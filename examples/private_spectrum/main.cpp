#include "Workflow.h"
#include <charconv>
#include <iostream>
#include <string_view>

namespace {
void usage(std::ostream &out) {
  out << "Usage: fhn-private-spectrum <plugin> [--scenario all|normal|local|shared]\n"
      << "                            [--frames N] [--no-scope-kernels] [--compare]\n"
      << "       fhn-private-spectrum [-h|--help]\n"
      << "\nSimulate a joint alarm for four operators, one sensor each, and eight complex bins.\n"
      << "  <plugin>            Native CKKS shared library (.dylib on macOS, .so on Linux).\n"
      << "  --scenario NAME     Run normal, local disturbance, shared oscillation,\n"
      << "                      or all three scenarios (default all).\n"
      << "  --frames N          Use 2..16 frames per window (default 4); each scenario\n"
      << "                      has a baseline and two observation windows.\n"
      << "  --no-scope-kernels  Skip span matching; keep scopes, prepared pair recipes,\n"
      << "                      and reference reuse across scopes.\n"
      << "  --compare           Verify and time all three paths: pair tiles, ordinary\n"
      << "                      pair recipes, and native ops with a shared conjugate.\n"
      << "  -h, --help          Show this help without loading the plugin.\n"
      << "\nDefault: scoped pair tiles. The trusted key-holder verifies every spectral bin\n"
      << "and releases only per-window/final alarms. The synthetic rule requires power\n"
      << "gain and coherent participation in the same bin for two consecutive windows.\n"
      << "This single-owner fixture is not distributed key management; CGGI is a TODO.\n";
}

unsigned frameCount(std::string_view input) {
  unsigned count = 0;
  const auto result = std::from_chars(input.data(), input.data() + input.size(), count);
  if (result.ec != std::errc{} || result.ptr != input.data() + input.size() || count < private_spectrum::minFrames ||
      count > private_spectrum::maxFrames)
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
    std::string scenario = "all";
    for (int i = 2; i < argc; ++i) {
      const std::string_view option = argv[i];
      if (option == "--no-scope-kernels")
        mode = Mode::Pair;
      else if (option == "--compare")
        compare = true;
      else if (option == "--frames" && i + 1 < argc)
        frames = frameCount(argv[++i]);
      else if (option == "--scenario" && i + 1 < argc) {
        scenario = argv[++i];
        if (scenario != "all" && scenario != "normal" && scenario != "local" && scenario != "shared")
          throw std::invalid_argument("--scenario requires all, normal, local, or shared");
      } else if (option == "-h" || option == "--help")
        help = true;
      else
        throw std::invalid_argument("unknown or incomplete option " + std::string(option));
    }
    if (help) {
      usage(std::cout);
      return 0;
    }
    run(std::string(plugin), frames, mode, compare, scenario);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "fhn-private-spectrum: " << error.what() << "\n";
    usage(std::cerr);
    return 1;
  }
}
