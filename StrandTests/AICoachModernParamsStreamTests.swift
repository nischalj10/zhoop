import Foundation
import XCTest
@testable import Strand

/// Answers successive requests from a queue and keeps each request's JSON body, so a test can see both
/// the first attempt and the resend.
private final class QueuedURLProtocolStub: URLProtocol {
    struct Stubbed { let status: Int; let body: String }
    nonisolated(unsafe) static var queue: [Stubbed] = []
    nonisolated(unsafe) static var bodies: [[String: Any]] = []

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.bodies.append(Self.jsonBody(of: request))
        let stub = Self.queue.isEmpty ? Stubbed(status: 500, body: "") : Self.queue.removeFirst()
        let response = HTTPURLResponse(
            url: request.url!, statusCode: stub.status, httpVersion: nil, headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(stub.body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    /// URLSession hands a protocol the body as a stream rather than as `httpBody`.
    private static func jsonBody(of request: URLRequest) -> [String: Any] {
        var data = request.httpBody ?? Data()
        if data.isEmpty, let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                data.append(buffer, count: count)
            }
        }
        return (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    static func session(_ responses: [Stubbed]) -> URLSession {
        queue = responses
        bodies = []
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [QueuedURLProtocolStub.self]
        return URLSession(configuration: configuration)
    }
}

/// Newer OpenAI chat and reasoning models reject `max_tokens` and `temperature` and want
/// `max_completion_tokens`. The non-streamed path always resent in that shape, but the Coach streams,
/// and the streamed path sent the classic shape once and surfaced the 400, so every question to such a
/// model failed. The rejection table is shared with Kotlin `AiCoachModernParamsTest`.
final class AICoachModernParamsStreamTests: XCTestCase {
    private let rejection = #"{"error":{"message":"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.","type":"invalid_request_error","param":"max_tokens","code":"unsupported_parameter"}}"#
    private let reply = "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n"
        + "data: {\"choices\":[{\"delta\":{\"content\":\"Rest \"}}]}\n\n"
        + "data: {\"choices\":[{\"delta\":{\"content\":\"well.\"}}]}\n\n"
        + "data: [DONE]\n\n"

    func testStreamResendsInTheModernShapeWhenTheModelAsksForIt() async throws {
        let session = QueuedURLProtocolStub.session([
            .init(status: 400, body: rejection),
            .init(status: 200, body: reply),
        ])
        var deltas: [String] = []
        try await OpenAIClient().stream(
            key: "k", model: "a-newer-model", systemPrompt: "s",
            messages: [(role: .user, content: "How did I sleep?")], session: session
        ) { deltas.append($0) }

        XCTAssertEqual(deltas.joined(), "Rest well.")
        let bodies = QueuedURLProtocolStub.bodies
        XCTAssertEqual(bodies.count, 2)
        XCTAssertEqual(bodies[0]["max_tokens"] as? Int, 4096)
        XCTAssertEqual(bodies[0]["temperature"] as? Double, 0.6)
        XCTAssertEqual(bodies[1]["max_completion_tokens"] as? Int, 4096)
        XCTAssertNil(bodies[1]["max_tokens"])
        XCTAssertNil(bodies[1]["temperature"])
        XCTAssertEqual(bodies[1]["stream"] as? Bool, true)
    }

    /// Any other 400 is about the request, not its shape: it surfaces after a single attempt.
    func testStreamDoesNotResendForAnUnrelated400() async {
        let session = QueuedURLProtocolStub.session([
            .init(status: 400, body: #"{"error":{"message":"The model `nope` does not exist."}}"#),
        ])
        do {
            try await OpenAIClient().stream(
                key: "k", model: "nope", systemPrompt: "s",
                messages: [(role: .user, content: "Hi")], session: session
            ) { _ in XCTFail("an error response must not emit a delta") }
            XCTFail("expected a server error")
        } catch let AICoachError.server(code, _) {
            XCTAssertEqual(code, 400)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        XCTAssertEqual(QueuedURLProtocolStub.bodies.count, 1)
    }

    func testWhichRejectionsAskForTheModernShape() {
        XCTAssertTrue(asksForModernOpenAIParams(
            "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."))
        XCTAssertTrue(asksForModernOpenAIParams(
            "Unsupported value: 'temperature' does not support 0.6 with this model."))
        XCTAssertFalse(asksForModernOpenAIParams("The model `nope` does not exist."))
        XCTAssertFalse(asksForModernOpenAIParams(""))
    }

    func testTheTwoParameterShapes() {
        let classic = openAIChatParams(modern: false)
        XCTAssertEqual(classic["max_tokens"] as? Int, 4096)
        XCTAssertEqual(classic["temperature"] as? Double, 0.6)
        XCTAssertNil(classic["max_completion_tokens"])

        let modern = openAIChatParams(modern: true)
        XCTAssertEqual(modern["max_completion_tokens"] as? Int, 4096)
        XCTAssertNil(modern["max_tokens"])
        XCTAssertNil(modern["temperature"])
    }
}
