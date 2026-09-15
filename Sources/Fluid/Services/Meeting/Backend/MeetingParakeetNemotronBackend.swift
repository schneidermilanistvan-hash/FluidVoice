import Foundation

// Stage E of `MEETING_TRANSCRIPTION_IMPLEMENTATION_PLAN.md`: the local composite backend.
// Parakeet TDT v2 produces word-timed ASR; Nemotron-3 produces per-epoch speaker activity. The
// backend owns orchestration — per-epoch materialization, unit construction, slot assignment and
// exactly-tiling coverage receipts — while the host-injected runtime owns the two model
// capabilities. It never touches the filesystem outside the frozen request's session directory,
// never merges speaker slots across tracks or epochs, and never invents timing: clamping happens
// only at physical epoch bounds, and text without word timings becomes one epoch-covering
// utterance rather than fabricated words.

@MainActor
final class MeetingParakeetNemotronBackend: MeetingTranscriptionBackend {
    static let descriptor = MeetingBackendDescriptor(
        id: .parakeetNemotron,
        version: "1",
        execution: .local,
        supportedLanguageCodes: ["en"],
        supportedTrackKinds: Set(MeetingAudioTrackKind.allCases),
        supportedFinalPrecisions: [.word, .utterance],
        resultContract: .canonicalEvidence,
        knownLimits: [
            "Nemotron-3 has 8 speaker slots per analysis epoch; a ninth voice is not reliably announced or separated.",
            "Speaker slots are epoch-scoped: no identity is merged across tracks or analysis epochs.",
            "English only; the fixed Parakeet TDT v2 meeting policy rejects other requested options.",
            "When ASR returns text without usable word timings, one utterance covering the epoch is emitted instead of fabricated words.",
            "Local Nemotron model must be installed before planning; execute performs no downloads.",
        ],
        analysisSampleRate: 16_000
    )

    var descriptor: MeetingBackendDescriptor {
        Self.descriptor
    }

    private let runtimeFactory: MeetingParakeetNemotronRuntimeFactory
    private let modelLocator: any MeetingNemotronModelLocating
    private let materializer: any MeetingEpochAudioMaterializing
    private var plannedArtifact: (attemptID: UUID, artifact: MeetingNemotronModelArtifact)?

    init(
        runtimeFactory: @escaping MeetingParakeetNemotronRuntimeFactory,
        modelLocator: any MeetingNemotronModelLocating = MeetingNemotronModelLocator(),
        materializer: any MeetingEpochAudioMaterializing = MeetingEpochAudioMaterializer()
    ) {
        self.runtimeFactory = runtimeFactory
        self.modelLocator = modelLocator
        self.materializer = materializer
    }

    func plan(_ request: MeetingBackendRequest) throws -> MeetingBackendPlan {
        guard self.descriptor.supportedLanguageCodes.contains(request.session.languageCode) else {
            throw MeetingBackendError.unsupportedLanguage(
                backend: self.descriptor.id,
                languageCode: request.session.languageCode
            )
        }
        // The fixed meeting Parakeet v2 policy rejects unsupported requested options explicitly
        // (plan §5): no coercion of another model or feature into this backend.
        _ = try MeetingProviderOptions.resolve(request.configuration)
        // Model readiness is a precondition of planning; execute performs no downloads.
        let artifact = try self.modelLocator.locate()
        let plan = MeetingBackendPlan(request: request, descriptor: self.descriptor)
        guard !plan.trackKindsByID.isEmpty else {
            throw MeetingBackendError.unsupportedTrackTopology(backend: self.descriptor.id)
        }
        self.plannedArtifact = (request.attemptID, artifact)
        return plan
    }

    func execute(
        plan: MeetingBackendPlan,
        manifest: MeetingAnalysisManifest?,
        progress: @escaping @MainActor (MeetingProcessingStage) -> Void
    ) async throws -> MeetingBackendOutcome {
        guard let manifest else {
            throw MeetingBackendError.outcomeContractMismatch(
                backend: self.descriptor.id,
                declared: .canonicalEvidence
            )
        }
        let validatedManifest = try manifest.validated(against: plan)
        guard let plannedArtifact = self.plannedArtifact,
              plannedArtifact.attemptID == plan.attemptID
        else {
            throw MeetingBackendError.planDisagreesWithRequest(
                backend: self.descriptor.id,
                defect: .attemptIdentity
            )
        }
        let artifact = try self.modelLocator.recheck(plannedArtifact.artifact)
        self.plannedArtifact = nil
        // The host hands out the runtime only for the exact frozen request.
        let runtime = try self.runtimeFactory(plan.request)
        let materializer = self.materializer
        let request = plan.request
        let work = Task.detached(priority: .userInitiated) {
            try await Self.runAttempt(
                request: request,
                manifest: validatedManifest,
                runtime: runtime,
                artifact: artifact,
                materializer: materializer,
                progress: progress
            )
        }
        let bundle = try await withTaskCancellationHandler {
            try await work.value
        } onCancel: {
            work.cancel()
        }
        try Task.checkCancellation()
        return .canonicalEvidence(bundle)
    }

    // MARK: - Orchestration (background context)

    /// One epoch's work item: its manifest spans in analysis order.
    private nonisolated struct EpochWork: Sendable {
        let track: MeetingAnalysisTrackManifest
        let epoch: MeetingAnalysisEpochRecord
        let spans: [MeetingAnalysisSpan]
    }

    /// Per-attempt cache for the PCM-first path. The cache is intentionally scoped to this
    /// detached attempt and bounded by the materializer's conservative sample limit; a future
    /// production implementation should replace it with a hashed immutable disk working store.
    private nonisolated final class MaterializationCache: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [MeetingAnalysisEpochID: MeetingMaterializedEpoch] = [:]
        private var totalSampleCount = 0

        func value(for epochID: MeetingAnalysisEpochID) -> MeetingMaterializedEpoch? {
            self.lock.lock(); defer { self.lock.unlock() }
            return self.values[epochID]
        }

        func insert(_ value: MeetingMaterializedEpoch, spanID: String) throws {
            self.lock.lock(); defer { self.lock.unlock() }
            let replacing = self.values[value.epochID]?.samples.count ?? 0
            let proposed = self.totalSampleCount - replacing + value.samples.count
            guard proposed <= MeetingEpochAudioMaterializer.conservativeSampleLimit else {
                throw MeetingEpochMaterializationError.sampleLimitExceeded(
                    spanID: spanID,
                    sampleCount: proposed
                )
            }
            self.values[value.epochID] = value
            self.totalSampleCount = proposed
        }
    }

    private nonisolated static func runAttempt(
        request: MeetingBackendRequest,
        manifest: MeetingAnalysisManifest,
        runtime: any MeetingParakeetNemotronRunning,
        artifact: MeetingNemotronModelArtifact,
        materializer: any MeetingEpochAudioMaterializing,
        progress: @escaping @MainActor (MeetingProcessingStage) -> Void
    ) async throws -> MeetingCanonicalResultBundle {
        let epochWork: [EpochWork] = manifest.tracks.flatMap { track in
            let spansByID = Dictionary(
                track.spans.map { ($0.id, $0) },
                uniquingKeysWith: { first, _ in first }
            )
            return track.epochs.map { epoch in
                EpochWork(
                    track: track,
                    epoch: epoch,
                    spans: epoch.spanIDs.compactMap { spansByID[$0] }
                )
            }
        }

        let pcmMaterializations = MaterializationCache()

        // Phase A: Nemotron diarization, one fresh state per epoch, then drained.
        await progress(.identifyingSpeakers)
        let phaseA = try await runtime.withNemotronDiarization(artifact: artifact) { factory in
            var result = MeetingNemotronPhaseResult()
            for work in epochWork {
                try Task.checkCancellation()
                let materialized: MeetingMaterializedEpoch
                do {
                    materialized = try await Self.materialize(
                        work: work,
                        manifest: manifest,
                        request: request,
                        materializer: materializer,
                        cache: pcmMaterializations
                    )
                } catch let error as CancellationError {
                    throw error
                } catch let error as MeetingEpochMaterializationError where error.isSampleLimitExceeded {
                    throw error
                } catch {
                    result.failures[work.epoch.id] = "epochMaterializationFailed"
                    continue
                }
                try Task.checkCancellation()
                do {
                    let diarizer = try await factory.makeDiarizer(epoch: work.epoch.id)
                    let segments = try await diarizer.diarize(samples: materialized.samples)
                    for segment in segments {
                        guard let start = Self.analysisTime(
                            forMaterializedSeconds: segment.start,
                            boundary: .start,
                            materialized: materialized,
                            spans: work.spans
                        ), let end = Self.analysisTime(
                            forMaterializedSeconds: segment.end,
                            boundary: .end,
                            materialized: materialized,
                            spans: work.spans
                        ) else { continue }
                        guard end > start else { continue }
                        result.activity.append(MeetingBackendSpeakerActivity(
                            token: MeetingBackendSpeakerToken(
                                analysisEpochID: work.epoch.id,
                                label: "slot-\(segment.slotIndex)"
                            ),
                            start: start,
                            end: end
                        ))
                    }
                } catch let error as CancellationError {
                    throw error
                } catch {
                    result.failures[work.epoch.id] = "diarizationFailed"
                    continue
                }
            }
            return result
        }
        try Task.checkCancellation()

        // Phase B: Parakeet ASR inside the attempt's single prepared-meeting scope. Epochs that
        // failed phase A are excluded from ASR as well: an epoch is one unit of work.
        await progress(.transcribing)
        let phaseB = try await runtime.withPreparedASR(
            attemptID: request.attemptID,
            configuration: request.configuration
        ) { asr in
            var result = MeetingParakeetPhaseResult()
            for work in epochWork where phaseA.failures[work.epoch.id] == nil {
                try Task.checkCancellation()
                let materialized: MeetingMaterializedEpoch
                do {
                    materialized = try await Self.materialize(
                        work: work,
                        manifest: manifest,
                        request: request,
                        materializer: materializer,
                        cache: pcmMaterializations
                    )
                } catch let error as CancellationError {
                    throw error
                } catch let error as MeetingEpochMaterializationError where error.isSampleLimitExceeded {
                    throw error
                } catch {
                    result.failures[work.epoch.id] = "epochMaterializationFailed"
                    continue
                }
                try Task.checkCancellation()
                do {
                    let output = try await asr.transcribeWithTimings(materialized.samples)
                    result.outputs.append(MeetingParakeetEpochOutput(
                        epochID: work.epoch.id,
                        text: output.result.text,
                        words: output.words,
                        sampleRate: materialized.sampleRate,
                        sampleCount: materialized.samples.count,
                        spanSamples: materialized.spanSamples
                    ))
                } catch let error as CancellationError {
                    throw error
                } catch {
                    result.failures[work.epoch.id] = "asrFailed"
                    continue
                }
            }
            return result
        }
        try Task.checkCancellation()

        return try self.assembleBundle(
            request: request,
            manifest: manifest,
            epochWork: epochWork,
            phaseA: phaseA,
            phaseB: phaseB
        )
    }

    private nonisolated static func materialize(
        work: EpochWork,
        manifest: MeetingAnalysisManifest,
        request: MeetingBackendRequest,
        materializer: any MeetingEpochAudioMaterializing,
        cache: MaterializationCache
    ) async throws -> MeetingMaterializedEpoch {
        let isPCMFirst = !work.spans.isEmpty && work.spans.allSatisfy {
            $0.chunk.analysisEncoding == .linearPCMFloat32CAFV1
        }
        if isPCMFirst, let cached = cache.value(for: work.epoch.id) {
            return cached
        }
        let materialized = try await materializer.materialize(
            epoch: work.epoch,
            track: work.track,
            manifest: manifest,
            sessionDirectory: request.sessionDirectory
        )
        try Task.checkCancellation()
        guard materialized.samples.count <= MeetingEpochAudioMaterializer.conservativeSampleLimit else {
            throw MeetingEpochMaterializationError.sampleLimitExceeded(
                spanID: work.spans.last?.id ?? "unknown",
                sampleCount: materialized.samples.count
            )
        }
        if isPCMFirst {
            try cache.insert(materialized, spanID: work.spans.last?.id ?? "unknown")
        }
        return materialized
    }

    // MARK: - Units, assignment and receipts

    /// Pure assembly of the final bundle: units from provider word timings clamped only at
    /// physical epoch bounds, slot assignment from overlapping epoch activity, and exactly one
    /// tiling receipt per admissible span. Failed epochs emit no units and mark every span failed;
    /// other epochs continue (plan §4).
    private nonisolated static func assembleBundle(
        request: MeetingBackendRequest,
        manifest: MeetingAnalysisManifest,
        epochWork: [EpochWork],
        phaseA: MeetingNemotronPhaseResult,
        phaseB: MeetingParakeetPhaseResult
    ) throws -> MeetingCanonicalResultBundle {
        let outputsByEpoch = Dictionary(
            phaseB.outputs.map { ($0.epochID, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let activityByEpoch = Dictionary(grouping: phaseA.activity, by: { $0.token.analysisEpochID })
        var failures = phaseA.failures
        failures.merge(phaseB.failures) { first, _ in first }

        var units: [MeetingFinalTextUnit] = []
        var receipts: [MeetingSpanCoverageReceipt] = []

        for work in epochWork {
            try Task.checkCancellation()
            if let reason = failures[work.epoch.id] {
                for span in work.spans {
                    receipts.append(Self.receipt(for: span, status: .failed, reasonCode: reason))
                }
                continue
            }
            for span in work.spans {
                receipts.append(Self.receipt(for: span, status: .processed))
            }
            guard let output = outputsByEpoch[work.epoch.id] else { continue }
            let epochActivity = activityByEpoch[work.epoch.id] ?? []
            units.append(contentsOf: Self.units(
                for: output,
                work: work,
                activity: epochActivity,
                attemptID: request.attemptID
            ))
        }

        return MeetingCanonicalResultBundle(
            evidence: MeetingFinalTranscriptEvidence(
                backendID: .parakeetNemotron,
                attemptID: request.attemptID,
                units: units,
                speakerActivity: phaseA.activity.filter { failures[$0.token.analysisEpochID] == nil }
            ),
            coverageReceipts: receipts
        )
    }

    private nonisolated static func receipt(
        for span: MeetingAnalysisSpan,
        status: MeetingSpanCoverageStatus,
        reasonCode: String? = nil
    ) -> MeetingSpanCoverageReceipt {
        MeetingSpanCoverageReceipt(
            id: "receipt:\(span.id)",
            spanID: span.id,
            analysisStart: span.analysisInterval.start,
            analysisEnd: span.analysisInterval.end,
            status: status,
            reasonCode: reasonCode
        )
    }

    /// Word-timed units for one successful epoch. Provider times are seconds into the epoch's
    /// materialized buffer, so analysis time is the epoch start plus the provider time, clamped
    /// only to the epoch's physical bounds. Words with unusable timings are dropped, never
    /// repaired; text with no usable word timings at all becomes one epoch-covering utterance.
    private nonisolated static func units(
        for output: MeetingParakeetEpochOutput,
        work: EpochWork,
        activity: [MeetingBackendSpeakerActivity],
        attemptID: UUID
    ) -> [MeetingFinalTextUnit] {
        let epochStart = work.epoch.analysisInterval.start
        let physicalEnd = Self.analysisTime(
            forMaterializedSeconds: Double(output.sampleCount) / output.sampleRate,
            boundary: .end,
            sampleRate: output.sampleRate,
            sampleCount: output.sampleCount,
            spanSamples: output.spanSamples,
            spans: work.spans
        ) ?? epochStart
        guard physicalEnd > epochStart else { return [] }

        var wordUnits: [MeetingFinalTextUnit] = []
        let sortedWords = output.words.enumerated()
            .sorted { ($0.element.start, $0.offset) < ($1.element.start, $1.offset) }
        for (index, word) in sortedWords {
            let text = word.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty,
                  word.start.isFinite, word.end.isFinite, word.end > word.start
            else { continue }
            guard let start = Self.analysisTime(
                forMaterializedSeconds: word.start,
                boundary: .start,
                sampleRate: output.sampleRate,
                sampleCount: output.sampleCount,
                spanSamples: output.spanSamples,
                spans: work.spans
            ), let end = Self.analysisTime(
                forMaterializedSeconds: word.end,
                boundary: .end,
                sampleRate: output.sampleRate,
                sampleCount: output.sampleCount,
                spanSamples: output.spanSamples,
                spans: work.spans
            ) else { continue }
            guard end > start else { continue }
            guard let spanIDs = Self.intersectingSpanIDs(start: start, end: end, spans: work.spans)
            else { continue }
            wordUnits.append(MeetingFinalTextUnit(
                id: "unit:\(attemptID.uuidString):\(work.epoch.id):\(index)",
                trackID: work.track.id,
                analysisEpochID: work.epoch.id,
                precision: .word,
                text: text,
                analysisStart: start,
                analysisEnd: end,
                speaker: Self.assignment(start: start, end: end, activity: activity),
                analysisSpanIDs: spanIDs,
                confidence: nil
            ))
        }
        if !wordUnits.isEmpty { return wordUnits }

        let text = output.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        let spanIDs = work.spans.map(\.id)
        guard !spanIDs.isEmpty else { return [] }
        return [MeetingFinalTextUnit(
            id: "unit:\(attemptID.uuidString):\(work.epoch.id):utterance",
            trackID: work.track.id,
            analysisEpochID: work.epoch.id,
            precision: .utterance,
            text: text,
            analysisStart: epochStart,
            analysisEnd: physicalEnd,
            speaker: Self.assignment(start: epochStart, end: physicalEnd, activity: activity),
            analysisSpanIDs: spanIDs,
            confidence: nil
        )]
    }

    /// The contiguous run of the epoch's spans intersecting the unit's analysis interval.
    /// Contiguity is structural: the epoch's spans tile its analysis interval with no holes.
    private nonisolated static func intersectingSpanIDs(
        start: TimeInterval,
        end: TimeInterval,
        spans: [MeetingAnalysisSpan]
    ) -> [String]? {
        let tolerance = MeetingAnalysisManifestSchema.mappingToleranceSeconds
        let ids = spans.filter {
            min($0.analysisInterval.end, end) - max($0.analysisInterval.start, start) > tolerance
        }.map(\.id)
        return ids.isEmpty ? nil : ids
    }

    private nonisolated enum MaterializedBoundary {
        case start
        case end
    }

    private nonisolated static func analysisTime(
        forMaterializedSeconds seconds: TimeInterval,
        boundary: MaterializedBoundary,
        materialized: MeetingMaterializedEpoch,
        spans: [MeetingAnalysisSpan]
    ) -> TimeInterval? {
        self.analysisTime(
            forMaterializedSeconds: seconds,
            boundary: boundary,
            sampleRate: materialized.sampleRate,
            sampleCount: materialized.samples.count,
            spanSamples: materialized.spanSamples,
            spans: spans
        )
    }

    /// Maps model/provider time through the actual resampled per-span sample ranges. This avoids
    /// accumulating a frame-rounding error at every source-file boundary and then pretending the
    /// concatenated buffer is an exact one-to-one copy of manifest analysis seconds.
    private nonisolated static func analysisTime(
        forMaterializedSeconds seconds: TimeInterval,
        boundary: MaterializedBoundary,
        sampleRate: Double,
        sampleCount: Int,
        spanSamples: [MeetingMaterializedSpanSamples],
        spans: [MeetingAnalysisSpan]
    ) -> TimeInterval? {
        guard seconds.isFinite, sampleRate.isFinite, sampleRate > 0, sampleCount > 0 else { return nil }
        let position = min(max(seconds * sampleRate, 0), Double(sampleCount))
        let probe: Double
        switch boundary {
        case .start:
            probe = min(position, Double(sampleCount).nextDown)
        case .end:
            probe = max(0, position == 0 ? 0 : position.nextDown)
        }
        let mappingsByID = Dictionary(
            spanSamples.map { ($0.spanID, $0.sampleRange) },
            uniquingKeysWith: { first, _ in first }
        )
        let spansByID = Dictionary(spans.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        guard let mapping = spanSamples.first(where: {
            Double($0.sampleRange.lowerBound) <= probe && probe < Double($0.sampleRange.upperBound)
        }), let range = mappingsByID[mapping.spanID], let span = spansByID[mapping.spanID], !range.isEmpty
        else { return nil }
        let fraction = min(max(
            (position - Double(range.lowerBound)) / Double(range.count),
            0
        ), 1)
        return span.analysisInterval.start + fraction * span.analysisInterval.duration
    }

    /// One overlapping slot assigns; two or more are reported ambiguous; none is unassigned.
    private nonisolated static func assignment(
        start: TimeInterval,
        end: TimeInterval,
        activity: [MeetingBackendSpeakerActivity]
    ) -> MeetingBackendSpeakerAssignment {
        let tolerance = MeetingAnalysisManifestSchema.mappingToleranceSeconds
        var tokens = Set<MeetingBackendSpeakerToken>()
        for entry in activity where min(entry.end, end) - max(entry.start, start) > tolerance {
            tokens.insert(entry.token)
        }
        let sorted = tokens.sorted { $0.label < $1.label }
        switch sorted.count {
        case 0: return .unassigned
        case 1: return .assigned(sorted[0])
        default: return .ambiguous(sorted)
        }
    }
}
