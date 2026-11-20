import AppKit
import SwiftUI

/// What the card's top line says. Numbers are shown only for `.live`.
enum RuntimeStatusCardStatus: Equatable {
    case notRunning
    case starting
    case reconnecting
    case live(RuntimeMetricsSnapshot)

    @MainActor
    init(store: RuntimeMetricsStore) {
        switch store.backendState {
        case .stopped, .failed, .stopping:
            self = .notRunning
        case .starting, .recovering:
            self = .starting
        case .running, .degraded:
            if !store.isStale, let snapshot = store.snapshot {
                self = .live(snapshot)
            } else {
                self = .reconnecting
            }
        }
    }

    /// The top line: the backend state, or the activity for live data. Shared with the HUD.
    func title(language: String) -> String {
        let key: GlanceText = switch self {
        case .notRunning: .backendNotRunning
        case .starting: .stateStarting
        case .reconnecting: .reconnecting
        case let .live(snapshot):
            switch snapshot.activity.state {
            case .generating: .stateGenerating
            case .prefilling: .statePrefilling
            case .loading: .stateLoading
            case .idle, .unknown: .stateIdle
            }
        }
        return GlanceText.localized(key, language: language)
    }
}

/// Text and proportions shown on the card, kept out of the view so they can be tested.
enum RuntimeStatusCardFormat {
    static let bytesPerGB: Double = 1_073_741_824

    /// "38.2" for bytes, in GB with one decimal.
    static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.1f", Double(max(0, bytes)) / bytesPerGB)
    }

    /// "38.2 / 96.0 GB", or "38.2 GB" when the ceiling is unknown.
    static func memoryUsage(_ memory: RuntimeMetricsSnapshot.Memory) -> String {
        guard memory.ceilingBytes > 0 else {
            return "\(gigabytes(memory.processBytes)) GB"
        }
        return "\(gigabytes(memory.processBytes)) / \(gigabytes(memory.ceilingBytes)) GB"
    }

    struct MemorySegments: Equatable {
        /// Process footprint minus KV and MLX cache, never below 0.
        var modelsAndOther: Int64
        var kv: Int64
        var cache: Int64
        /// The bar's full width: the ceiling, or the segments' sum when the ceiling is unknown or smaller.
        var total: Int64

        func fraction(_ bytes: Int64) -> Double {
            total > 0 ? Double(bytes) / Double(total) : 0
        }
    }

    static func memorySegments(_ memory: RuntimeMetricsSnapshot.Memory) -> MemorySegments {
        let kv = max(0, memory.kvCacheBytes)
        let cache = max(0, memory.mlxCacheBytes)
        let models = max(0, memory.processBytes - kv - cache)
        return MemorySegments(
            modelsAndOther: models,
            kv: kv,
            cache: cache,
            total: max(memory.ceilingBytes, models + kv + cache)
        )
    }

    /// "87%", or "—" before any request could use the prefix cache.
    static func prefixHit(_ prefix: RuntimeMetricsSnapshot.PrefixCache) -> String {
        guard prefix.eligibleTokens > 0 else {
            return "—"
        }
        let ratio = Double(max(0, prefix.hitTokens)) / Double(prefix.eligibleTokens)
        return "\(Int((min(1, ratio) * 100).rounded()))%"
    }

    /// "2": requests in progress. No denominator: queue capacity is the admission limit, not how
    /// many requests run at once. Shared with the HUD.
    static func active(_ snapshot: RuntimeMetricsSnapshot) -> String {
        "\(snapshot.activity.activeRequests)"
    }

    /// The model the card names: the one serving requests, else the default, else the first.
    static func focusModel(_ models: [RuntimeMetricsSnapshot.Model]) -> RuntimeMetricsSnapshot.Model? {
        models.first { $0.activeRequests > 0 } ?? models.first(where: \.isDefault) ?? models.first
    }

    /// "Qwen3.6-35B-A3B-4bit · DFlash2"; the mode is left out for normal decoding.
    static func modelLine(_ models: [RuntimeMetricsSnapshot.Model]) -> String? {
        guard let model = focusModel(models), !model.id.isEmpty else {
            return nil
        }
        let name = model.id.split(separator: "/").last.map(String.init) ?? model.id
        let mode: String? = switch model.mode?.lowercased() {
        case "dflash2": "DFlash2"
        case "mtp": "MTP"
        case nil, "", "normal": nil
        case let other?: other
        }
        let extra = models.count > 1 ? " +\(models.count - 1)" : ""
        return [name + extra, mode].compactMap { $0 }.joined(separator: " · ")
    }

    /// The degraded reason that explains `indicator`'s color while health is not healthy, as a
    /// short title. Shared by the card and both HUD densities.
    static func alertReason(
        _ snapshot: RuntimeMetricsSnapshot, indicator: GlanceIndicator?, language: String
    ) -> String? {
        guard snapshot.health.status == .degraded || snapshot.health.status == .down,
              let reason = alertReasonCode(snapshot.health.degradedReasons, indicator: indicator)
        else {
            return nil
        }
        return GlanceText.reasonTitle(reason, language: language)
    }

    /// Picks the reason behind the indicator's color. The backend lists scheduler reasons by
    /// declaration (model failure last) and appends memory reasons, so the first one may be a
    /// mild reason next to a red cause. Red prefers its red causes, amber prefers memory pressure;
    /// otherwise received order, with reasons this App does not know last.
    static func alertReasonCode(_ reasons: [String], indicator: GlanceIndicator?) -> String? {
        let preferred: [String] = switch indicator {
        case .red: ["process_memory_emergency", "model_failed", "process_memory_hard"]
        case .amber: ["process_memory_hard", "process_memory_soft"]
        case .blue, .green, nil: []
        }
        if let match = preferred.first(where: reasons.contains) {
            return match
        }
        return reasons.first(where: GlanceText.isKnownReason) ?? reasons.first
    }

    static func speed(_ value: Double?) -> String {
        guard let value, value.isFinite else {
            return "—"
        }
        return String(format: "%.1f", value)
    }

    static func tokensPerSecond(_ value: Double?) -> String {
        guard let value, value.isFinite else {
            return "—"
        }
        return "\(Int(value.rounded()).formatted(.number.grouping(.automatic))) tok/s"
    }

    static func milliseconds(_ value: Double?) -> String {
        guard let value, value.isFinite else {
            return "—"
        }
        return "\(Int(value.rounded())) ms"
    }

    static func uptime(_ seconds: TimeInterval, language: String) -> String {
        let formatter = DateComponentsFormatter()
        formatter.unitsStyle = .abbreviated
        formatter.maximumUnitCount = 2
        formatter.allowedUnits = seconds < 60 ? [.second] : [.day, .hour, .minute]
        var calendar = Calendar(identifier: .gregorian)
        calendar.locale = Locale(identifier: localeIdentifier(language))
        formatter.calendar = calendar
        let duration = formatter.string(from: seconds) ?? ""
        return String(format: GlanceText.localized(.uptime, language: language), duration)
    }

    private static func localeIdentifier(_ language: String) -> String {
        switch MenuBarMenuBuilder.normalizedLanguage(language) {
        case "zh": "zh-Hans"
        case "zh-Hant": "zh-Hant"
        case "ja": "ja"
        case "ko": "ko"
        default: "en"
        }
    }
}

/// The live status card at the top of the menu bar menu (spec 5.5). It reads the store
/// directly, so an open menu updates through Observation without being rebuilt.
struct RuntimeStatusCardView: View {
    static let width: CGFloat = 310

    let store: RuntimeMetricsStore
    let language: String
    /// Called with the card's height whenever it changes, so the hosting menu item can resize.
    var onHeightChange: ((CGFloat) -> Void)?

    var body: some View {
        content
            .padding(12)
            .frame(width: Self.width, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
            .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height in
                onHeightChange?(height)
            }
    }

    @ViewBuilder
    private var content: some View {
        let status = RuntimeStatusCardStatus(store: store)
        VStack(alignment: .leading, spacing: 0) {
            header(status)
            if case let .live(snapshot) = status {
                if let line = RuntimeStatusCardFormat.modelLine(snapshot.models) {
                    Text(line)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .padding(.top, 4)
                }
                performancePanel(snapshot)
                    .padding(.top, 12)
                memorySection(snapshot)
                    .padding(.top, 12)
            }
        }
    }

    private func header(_ status: RuntimeStatusCardStatus) -> some View {
        HStack(spacing: 8) {
            Circle()
                .fill(dotColor)
                .frame(width: 8, height: 8)
            Text(title(status))
                .font(.system(size: 13, weight: .medium))
                .lineLimit(1)
            if case let .live(snapshot) = status, let reason = alertReason(snapshot) {
                Text(reason)
                    .font(.system(size: 12))
                    .foregroundStyle(dotColor)
                    .lineLimit(1)
            }
            Spacer(minLength: 8)
            if case .live = status {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    if let uptime = store.uptime(now: context.date) {
                        Text(RuntimeStatusCardFormat.uptime(uptime, language: language))
                            .font(.system(size: 11))
                            .foregroundStyle(.tertiary)
                            .monospacedDigit()
                    }
                }
            }
        }
    }

    private func performancePanel(_ snapshot: RuntimeMetricsSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(RuntimeStatusCardFormat.speed(snapshot.performance.liveDecodeTokensPerSecond))
                    .font(.system(size: 26, weight: .medium))
                    .monospacedDigit()
                Text(text(.decodeUnit))
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                Text(text(.window60))
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
            RuntimeSparklineShape(values: store.sparkline.values)
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
                .frame(height: 40)
            // Two columns as in the mockup; one column when a language's labels do not fit two-up.
            ViewThatFits(in: .horizontal) {
                metricsGrid(snapshot)
                VStack(spacing: 4) {
                    ForEach(metrics(snapshot), id: \.label) { metric in
                        row(metric.label, metric.value)
                    }
                }
            }
        }
        .padding(EdgeInsets(top: 10, leading: 10, bottom: 8, trailing: 10))
        .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
    }

    private func memorySection(_ snapshot: RuntimeMetricsSnapshot) -> some View {
        let memory = snapshot.memory
        let segments = RuntimeStatusCardFormat.memorySegments(memory)
        return VStack(alignment: .leading, spacing: 6) {
            row(text(.memory), RuntimeStatusCardFormat.memoryUsage(memory))
            RuntimeMemoryBar(segments: segments, height: 8)
            // Longer translations drop the repeated "GB" rather than truncating a label.
            ViewThatFits(in: .horizontal) {
                legendRow(segments, unit: " GB")
                legendRow(segments, unit: "")
            }
            row(text(.prefixHit), RuntimeStatusCardFormat.prefixHit(snapshot.prefixCache))
                .padding(.top, 2)
        }
    }

    static let modelsColor = Color.blue
    static let kvColor = Color.purple
    static let cacheColor = Color.primary.opacity(0.35)

    private struct Metric {
        var label: String
        var value: String
    }

    private func metrics(_ snapshot: RuntimeMetricsSnapshot) -> [Metric] {
        [
            Metric(
                label: text(.prefill),
                value: RuntimeStatusCardFormat.tokensPerSecond(snapshot.performance.prefillTokensPerSecond)
            ),
            Metric(label: text(.ttft), value: RuntimeStatusCardFormat.milliseconds(snapshot.performance.ttftMs)),
            Metric(label: text(.active), value: RuntimeStatusCardFormat.active(snapshot)),
            Metric(label: text(.queued), value: "\(snapshot.activity.queuedRequests)"),
        ]
    }

    /// The 2 × 2 grid; labels keep their full width so `ViewThatFits` can reject it when too wide.
    private func metricsGrid(_ snapshot: RuntimeMetricsSnapshot) -> some View {
        let metrics = metrics(snapshot)
        return Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
            ForEach([0, 2], id: \.self) { start in
                GridRow {
                    ForEach(metrics[start..<start + 2], id: \.label) { metric in
                        HStack(spacing: 8) {
                            Text(metric.label)
                                .foregroundStyle(.secondary)
                                .fixedSize()
                            Spacer(minLength: 4)
                            Text(metric.value)
                                .monospacedDigit()
                                .fixedSize()
                        }
                        .font(.system(size: 12))
                    }
                }
            }
        }
    }

    /// A label and its value; the value always shows in full and a long label shrinks first.
    private func row(_ label: String, _ value: String) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .foregroundStyle(.secondary)
                .minimumScaleFactor(0.7)
            Spacer(minLength: 4)
            Text(value)
                .monospacedDigit()
                .fixedSize()
                .layoutPriority(1)
        }
        .font(.system(size: 12))
        .lineLimit(1)
    }

    private func legendRow(_ segments: RuntimeStatusCardFormat.MemorySegments, unit: String) -> some View {
        HStack(spacing: 12) {
            legend(Self.modelsColor, .memoryModels, segments.modelsAndOther, unit: unit)
            legend(Self.kvColor, .memoryKV, segments.kv, unit: unit)
            legend(Self.cacheColor, .memoryCache, segments.cache, unit: unit)
        }
    }

    private func legend(_ color: Color, _ key: GlanceText, _ bytes: Int64, unit: String) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 1.5)
                .fill(color)
                .frame(width: 7, height: 7)
            Text("\(text(key)) \(RuntimeStatusCardFormat.gigabytes(bytes))\(unit)")
                .monospacedDigit()
        }
        .font(.system(size: 11))
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .fixedSize()
    }

    private var dotColor: Color {
        Self.dotColor(for: store.indicator)
    }

    /// The status dot's color: the menu bar dot's color, or a muted gray when there is no dot.
    static func dotColor(for indicator: GlanceIndicator?) -> Color {
        guard let indicator else {
            return Color.secondary.opacity(0.6)
        }
        return Color(nsColor: MenuBarStatusDotView.color(for: indicator))
    }

    private func title(_ status: RuntimeStatusCardStatus) -> String {
        status.title(language: language)
    }

    /// The reason behind the dot's color, shown next to the state while health is not healthy.
    private func alertReason(_ snapshot: RuntimeMetricsSnapshot) -> String? {
        RuntimeStatusCardFormat.alertReason(snapshot, indicator: store.indicator, language: language)
    }

    private func text(_ key: GlanceText) -> String {
        GlanceText.localized(key, language: language)
    }
}

/// The memory bar: models and other, KV and MLX cache, against the ceiling. Shared with the HUD.
struct RuntimeMemoryBar: View {
    var segments: RuntimeStatusCardFormat.MemorySegments
    var height: CGFloat

    var body: some View {
        GeometryReader { proxy in
            HStack(spacing: 0) {
                Rectangle().fill(RuntimeStatusCardView.modelsColor)
                    .frame(width: proxy.size.width * segments.fraction(segments.modelsAndOther))
                Rectangle().fill(RuntimeStatusCardView.kvColor)
                    .frame(width: proxy.size.width * segments.fraction(segments.kv))
                Rectangle().fill(RuntimeStatusCardView.cacheColor)
                    .frame(width: proxy.size.width * segments.fraction(segments.cache))
                Spacer(minLength: 0)
            }
        }
        .frame(height: height)
        .background(Color.primary.opacity(0.07))
        .clipShape(RoundedRectangle(cornerRadius: height / 2))
    }
}

/// The 60-second decode speed line: newest sample on the right edge, gaps where no speed was known.
struct RuntimeSparklineShape: Shape {
    var values: [Double?]

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let capacity = RuntimeSparkline.capacity
        guard rect.width > 0, rect.height > 0, !values.isEmpty else {
            return path
        }
        let peak = max(values.compactMap { $0 }.filter(\.isFinite).max() ?? 0, 1)
        let step = rect.width / CGFloat(capacity - 1)
        let offset = capacity - values.count
        let inset: CGFloat = 2
        var drawing = false
        for (index, value) in values.enumerated() {
            guard let value, value.isFinite else {
                drawing = false
                continue
            }
            let point = CGPoint(
                x: rect.minX + CGFloat(offset + index) * step,
                y: rect.maxY - inset - CGFloat(min(max(value, 0) / peak, 1)) * (rect.height - inset * 2)
            )
            if drawing {
                path.addLine(to: point)
            } else {
                path.move(to: point)
                drawing = true
            }
        }
        return path
    }
}

extension RuntimeStatusCardView {
    /// The view for the card's menu item: created once per menu build, then kept current by
    /// Observation, and resized when the card's height changes (for example when the backend stops).
    @MainActor
    static func makeMenuItemView(store: RuntimeMetricsStore, language: String) -> NSView {
        let hosting = NSHostingView(rootView: RuntimeStatusCardView(store: store, language: language))
        hosting.rootView.onHeightChange = { [weak hosting] height in
            guard let hosting, height > 0, hosting.frame.height != height else {
                return
            }
            hosting.setFrameSize(NSSize(width: width, height: height))
        }
        refit(hosting)
        return hosting
    }

    /// Sizes the card's menu item view to its current content: 310 pt wide, fitting height.
    @MainActor
    static func refit(_ view: NSView) {
        view.layoutSubtreeIfNeeded()
        let size = NSSize(width: width, height: max(1, view.fittingSize.height))
        if view.frame.size != size {
            view.setFrameSize(size)
        }
    }
}
