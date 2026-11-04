#include "Workflow.h"
#include <charconv>
#include <iostream>
#include <string_view>

namespace {
void usage(std::ostream &out) {
  out << "Usage: fhn-private-spectrum <plugin> [--frames N] [--no-scope-kernels] [--compare]\n"
      << "       fhn-private-spectrum [-h|--help]\n"
      << "\nAccumulate encrypted cross/power spectra for four sensors and eight complex bins.\n"
      << "  <plugin>            Native CKKS shared library (.dylib on macOS, .so on Linux).\n"
      << "  --frames N          Use 2..16 synthetic synchronized frames (default 4).\n"
      << "  --no-scope-kernels  Skip span matching; keep scopes, prepared pair recipes,\n"
      << "                      and reference reuse across scopes.\n"
      << "  --compare           Verify and time all three paths: pair tiles, ordinary\n"
      << "                      pair recipes, and native ops with a shared conjugate.\n"
      << "  -h, --help          Show this help without loading the plugin.\n"
      << "\nDefault: scoped pair tiles. Every run checks all decrypted complex bins\n"
      << "against a cleartext oracle. Client phase/coherence uses the finite-window sums.\n";
}

unsigned frameCount(std::string_view input) {
  unsigned count = 0;
  const auto result = std::from_chars(input.data(), input.data() + input.size(), count);
  if (result.ec != std::errc{} || result.ptr != input.data() + input.size() ||
      count < private_spectrum::minFrames || count > private_spectrum::maxFrames)
    throw std::invalid_argument("--frames requires an integer from 2 to 16");
  return count;
}
} // namespace

int main(int argc, char **argv) {
  using namespace private_spectrum;
  try {
    if (argc < 2)
      throw std::invalid_argument("missing CKKS plugin path");
    const std::string_view plugin = argv[1];
    if (argc == 2 && (plugin == "--help" || plugin == "-h")) {
      usage(std::cout);
      return 0;
    }
    if (plugin.empty() || plugin.front() == '-')
      throw std::invalid_argument("expected CKKS plugin path");

    unsigned frames = 4;
    auto mode = Mode::Tile;
    bool compare = false;
    bool help = false;
    for (int i = 2; i < argc; ++i) {
      const std::string_view option = argv[i];
      if (option == "--no-scope-kernels")
        mode = Mode::Pair;
      else if (option == "--compare")
        compare = true;
      else if (option == "--frames" && i + 1 < argc)
        frames = frameCount(argv[++i]);
      else if (option == "-h" || option == "--help")
        help = true;
      else
        throw std::invalid_argument("unknown or incomplete option " + std::string(option));
    }
    if (help) {
      usage(std::cout);
      return 0;
    }
    run(std::string(plugin), frames, mode, compare);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "fhn-private-spectrum: " << error.what() << "\n";
    usage(std::cerr);
    return 1;
  }
}
