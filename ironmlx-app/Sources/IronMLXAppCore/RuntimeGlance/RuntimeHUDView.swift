import SwiftUI

/// The floating HUD's content (spec 5.6, mockup "悬浮 HUD"). Reads the store and preferences
/// directly, so it updates through Observation without the controller pushing values. The
/// context menu is AppKit (`RuntimeHUDContextMenu`), opened by the panel.
struct RuntimeHUDView: View {
    static let standardWidth: CGFloat = 220
    static let cornerRadius: CGFloat = 12

    let store: RuntimeMetricsStore
    let preferences: GlancePreferences
    var language: String
    /// Called with the content's size whenever it changes, so the panel can follow it.
    var onSizeChange: ((CGSize) -> Void)?

    @State private var isHovering = false

    var body: some View {
        let status = RuntimeStatusCardStatus(store: store)
        let alertColor = alertColor
        content(status, alertColor: alertColor)
            .glassEffect(.regular, in: RoundedRectangle(cornerRadius: Self.cornerRadius))
            .overlay {
                RoundedRectangle(cornerRadius: Self.cornerRadius)
                    .strokeBorder(
                        alertColor ?? Color.primary.opacity(0.18),
                        lineWidth: alertColor == nil ? 0.5 : 1
                    )
            }
            .opacity(isHovering ? 1 : 0.85)
            .onHover { isHovering = $0 }
            .fixedSize()
            .onGeometryChange(for: CGSize.self) { $0.size } action: { size in
                onSizeChange?(size)
            }
    }

    @ViewBuilder
    private func content(_ status: RuntimeStatusCardStatus, alertColor: Color?) -> some View {
        switch preferences.hudDensity {
        case .mini:
            mini(status, alertColor: alertColor)
                .padding(EdgeInsets(top: 6, leading: 10, bottom: 6, trailing: 10))
        case .standard:
            standard(status, alertColor: alertColor)
                .padding(EdgeInsets(top: 10, leading: 12, bottom: 10, trailing: 12))
                .frame(width: Self.standardWidth, alignment: .leading)
        }
    }

    // MARK: Mini: dot, "42.8 t/s", "· 2 Active"

    @ViewBuilder
    private func mini(_ status: RuntimeStatusCardStatus, alertColor: Color?) -> some View {
        HStack(spacing: 8) {
            marker(alertColor: alertColor)
            if case let .live(snapshot) = status {
                Text("\(RuntimeStatusCardFormat.speed(snapshot.performance.liveDecodeTokensPerSecond)) t/s")
                    .font(.system(size: 12).monospacedDigit())
                if let alertColor {
                    Text("· \(title(status))")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(alertColor)
                } else {
                    Text("· \(RuntimeStatusCardFormat.active(snapshot)) \(text(.active))")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            } else {
                Text(title(status))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(alertColor ?? .primary)
            }
        }
        .lineLimit(1)
    }

    // MARK: Standard: state and model, speed and trend, TTFT / active / queued, memory

    @ViewBuilder
    private func standard(_ status: RuntimeStatusCardStatus, alertColor: Color?) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                marker(alertColor: alertColor)
                Text(title(status))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(alertColor ?? .primary)
                    .lineLimit(1)
                    .layoutPriority(1)
                Spacer(minLength: 6)
                if case let .live(snapshot) = status, let model = RuntimeStatusCardFormat.modelLine(snapshot.models) {
                    Text(model)
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            if case let .live(snapshot) = status {
                live(snapshot)
            }
        }
    }

    @ViewBuilder
    private func live(_ snapshot: RuntimeMetricsSnapshot) -> some View {
        HStack(alignment: .center, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 5) {
                Text(RuntimeStatusCardFormat.speed(snapshot.performance.liveDecodeTokensPerSecond))
                    .font(.system(size: 22, weight: .medium).monospacedDigit())
                Text("tok/s")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 6)
            RuntimeSparklineShape(values: store.sparkline.values)
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
                .frame(width: 80, height: 22)
        }
        .padding(.top, 6)

        let metrics = [
            (text(.ttft), RuntimeStatusCardFormat.milliseconds(snapshot.performance.ttftMs)),
            (text(.active), RuntimeStatusCardFormat.active(snapshot)),
            (text(.queued), "\(snapshot.activity.queuedRequests)"),
        ]
        // One line as in the mockup; one metric per line when a language's labels do not fit.
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 0) {
                ForEach(Array(metrics.enumerated()), id: \.offset) { index, metric in
                    if index > 0 {
                        Spacer(minLength: 8)
                    }
                    metricPair(metric.0, metric.1)
                }
            }
            VStack(alignment: .leading, spacing: 2) {
                ForEach(metrics, id: \.0) { metric in
                    HStack {
                        metricPair(metric.0, nil)
                        Spacer(minLength: 4)
                        Text(metric.1).monospacedDigit().foregroundStyle(.primary)
                    }
                    .font(.system(size: 11))
                }
            }
        }
        .padding(.top, 4)

        RuntimeMemoryBar(segments: RuntimeStatusCardFormat.memorySegments(snapshot.memory), height: 5)
            .padding(.top, 8)
        HStack(spacing: 8) {
            Text(text(.memory))
                .foregroundStyle(.secondary)
            Spacer(minLength: 4)
            Text(RuntimeStatusCardFormat.memoryUsage(snapshot.memory))
                .monospacedDigit()
        }
        .font(.system(size: 12))
        .lineLimit(1)
        .padding(.top, 4)
    }

    private func metricPair(_ label: String, _ value: String?) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .foregroundStyle(.secondary)
            if let value {
                Text(value)
                    .monospacedDigit()
            }
        }
        .font(.system(size: 11))
        .lineLimit(1)
        .fixedSize()
    }

    // MARK: Shared pieces

    /// The status dot, or a warning sign in the alert color. Static: the alert style never blinks.
    @ViewBuilder
    private func marker(alertColor: Color?) -> some View {
        if let alertColor {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 10))
                .foregroundStyle(alertColor)
        } else {
            Circle()
                .fill(RuntimeStatusCardView.dotColor(for: store.indicator))
                .frame(width: 7, height: 7)
        }
    }

    /// Red or amber per `HUDVisibilityPolicy.isAlert`; nil when not alerting (including a
    /// backend the user stopped, which keeps its red dot but not the alert style).
    private var alertColor: Color? {
        guard HUDVisibilityPolicy.isAlert(indicator: store.indicator, backendState: store.backendState) else {
            return nil
        }
        return RuntimeStatusCardView.dotColor(for: store.indicator)
    }

    /// The reason behind the alert color while alerting, otherwise the state.
    private func title(_ status: RuntimeStatusCardStatus) -> String {
        if alertColor != nil, case let .live(snapshot) = status,
           let reason = RuntimeStatusCardFormat.alertReason(snapshot, indicator: store.indicator, language: language)
        {
            return reason
        }
        return status.title(language: language)
    }

    private func text(_ key: GlanceText) -> String {
        GlanceText.localized(key, language: language)
    }
}
