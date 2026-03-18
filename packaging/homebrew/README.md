# Homebrew Installation

## Quick Install (after tap is published)

```bash
brew tap apepkuss/ironmlx
brew install ironmlx
```

## Local Development Install

```bash
brew install --formula ./packaging/homebrew/ironmlx.rb
```

## Usage

### Start manually
```bash
ironmlx --model mlx-community/Qwen3-0.6B-4bit --port 8080
```

### Run as background service
```bash
brew services start ironmlx
```

### Stop service
```bash
brew services stop ironmlx
```

### Web Admin
Open http://localhost:8080/admin in your browser.
