class Ironmlx < Formula
  desc "Local LLM inference engine for Apple Silicon, powered by MLX"
  homepage "https://github.com/apepkuss/ironmlx"
  url "https://github.com/apepkuss/ironmlx/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "rust" => :build
  depends_on "cmake" => :build
  depends_on :macos

  def install
    system "cargo", "build", "--release", "--bin", "ironmlx"
    bin.install "target/release/ironmlx"
  end

  def post_install
    (var/"log/ironmlx").mkpath
  end

  service do
    run [opt_bin/"ironmlx", "--port", "8080"]
    keep_alive true
    log_path var/"log/ironmlx/output.log"
    error_log_path var/"log/ironmlx/error.log"
    working_dir HOMEBREW_PREFIX
  end

  def caveats
    <<~EOS
      ironmlx requires Apple Silicon (M1 or later) with Metal support.

      To start the server with a model:
        ironmlx --model <model-path-or-hf-repo> --port 8080

      To run as a background service:
        brew services start ironmlx

      Web Admin Panel:
        http://localhost:8080/admin

      API Endpoints:
        http://localhost:8080/v1/chat/completions (OpenAI compatible)
        http://localhost:8080/v1/messages (Anthropic compatible)
    EOS
  end

  test do
    assert_match "ironmlx", shell_output("#{bin}/ironmlx --help 2>&1", 1)
  end
end
