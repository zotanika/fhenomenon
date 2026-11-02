// Diagnostic harness: oracle, timing, memory accounting and four execution paths.
// Start at main.cpp for the application; ScopedSearch/EncryptedSearch describe the two request paths.
#include "FHN/contrib/SearchKernels.h"
#include "Fixture.h"
#include "ScopedSearch.h"
#include "Session/Session.h"
#include "Usage.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>

namespace {
using namespace private_search;
using namespace fhenomenon::runtime;
using fhenomenon::contrib::SearchShape;
using nlohmann::json;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::uint64_t payloadBytes(const NativeServices &services, NativeRuntime &native, const Value &value) {
  FhnCkksBufferInfo info{};
  require(services.buffer_info(services.context.get(), native.buffer(value), &info) == 0,
          "cannot inspect ciphertext payload");
  return info.live_payload_bytes;
}
unsigned positive(const std::string &text, unsigned maximum) {
  require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos, "expected positive integer");
  auto value = std::stoull(text);
  require(value > 0 && value <= maximum, "integer option out of range");
  return static_cast<unsigned>(value);
}
struct Options {
  std::string library, fixture, query, output;
  unsigned repetitions = 3, limit = 0;
  bool compare = false, all = false, scoped = false, scope_kernels = true;
};
Options options(int argc, char **argv) {
  require(argc >= 3, benchmark_usage);
  Options result;
  result.library = argv[1];
  result.fixture = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--compare")
      result.compare = true;
    else if (flag == "--verify-all")
      result.all = true;
    else if (flag == "--scoped")
      result.scoped = true;
    else if (flag == "--no-scope-kernels")
      result.scope_kernels = false;
    else {
      require(i + 1 < argc, "missing value for " + flag);
      const std::string value = argv[++i];
      if (flag == "--query")
        result.query = value;
      else if (flag == "--json")
        result.output = value;
      else if (flag == "--reps")
        result.repetitions = positive(value, 1000);
      else if (flag == "--limit-documents")
        result.limit = positive(value, 100000);
      else
        throw std::runtime_error("unknown option: " + flag);
    }
  }
  require(result.scoped || result.scope_kernels, "--no-scope-kernels requires --scoped");
  require(!result.all || result.query.empty(), "--query and --verify-all are mutually exclusive");
  if (!result.output.empty()) {
    std::error_code error;
    const bool same = std::filesystem::equivalent(result.fixture, result.output, error);
    require(!same &&
              std::filesystem::weakly_canonical(result.fixture) != std::filesystem::weakly_canonical(result.output),
            "JSON output must not overwrite the input fixture");
  }
  return result;
}

struct Run {
  Runtime runtime{RuntimeOptions{0, 8192}};
  NativeRuntime native;
  Value index;
  EncryptedDocuments documents;
  std::shared_ptr<fhenomenon::Session> session;
  unsigned capacity;
  bool prepared;
  std::string name;
  double index_setup_ms;
  std::size_t corpus_bytes, corpus_ciphertext_bytes = 0, corpus_ciphertext_reserved_bytes = 0;
  Run(TrustedClient &client, const Options &opts, const Fixture &fixture, unsigned cap, bool prep)
    : native(runtime, client.computeServices()), capacity(cap), prepared(prep),
      name(std::string(cap == 1 ? "row" : "tile") + (prep ? "-prepared" : "-ordinary")) {
    fhenomenon::contrib::registerSearchKernels(runtime, client.computeServices(), opts.library);
    const auto start = Clock::now();
    Values blocks;
    for (size_t first = 0; first < fixture.documents.size(); first += capacity) {
      std::vector<double> packed(slots);
      for (unsigned lane = 0; lane < tile_capacity; ++lane) {
        const size_t document = first + (capacity == 1 ? 0 : lane);
        if (document < fixture.documents.size()) {
          const auto &v = fixture.documents[document].embedding;
          std::copy(v.begin(), v.end(), packed.begin() + lane * width);
        }
      }
      blocks.push_back(client.encryptVector(native, packed));
      corpus_ciphertext_bytes += payloadBytes(client.computeServices(), native, blocks.back());
      corpus_ciphertext_reserved_bytes += blocks.back().bytes();
    }
    index = runtime
              .submit({"contrib.search.index", blocks,
                       SearchShape{width, static_cast<uint32_t>(fixture.documents.size()), capacity}})
              .at(0);
    index_setup_ms = elapsed(start);
    documents = {std::move(blocks), SearchShape{width, static_cast<uint32_t>(fixture.documents.size()), capacity}};
    if (opts.scoped) {
      ScopeOptions scope_options;
      scope_options.use_kernels = opts.scope_kernels;
      session = fhenomenon::Session::create(runtime, scope_options);
    }
    corpus_bytes = runtime.stats().live_bytes;
  }
};

std::vector<size_t> rank(const Fixture &fixture, const std::vector<double> &scores) {
  std::vector<size_t> result(scores.size());
  std::iota(result.begin(), result.end(), 0);
  std::sort(result.begin(), result.end(), [&](size_t a, size_t b) {
    return scores[a] != scores[b] ? scores[a] > scores[b] : fixture.documents[a].id < fixture.documents[b].id;
  });
  return result;
}
std::vector<double> oracle(const Fixture &fixture, const Query &query) {
  std::vector<double> scores;
  for (const auto &document : fixture.documents)
    scores.push_back(
      std::inner_product(query.embedding.begin(), query.embedding.end(), document.embedding.begin(), 0.0));
  return scores;
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}
json search(TrustedClient &client, Run &run, const Fixture &fixture, const Query &query, unsigned repetitions) {
  auto start = Clock::now();
  std::vector<double> packed(slots);
  for (unsigned lane = 0; lane < tile_capacity; ++lane)
    std::copy(query.embedding.begin(), query.embedding.end(), packed.begin() + lane * width);
  auto encrypted_query = client.encryptVector(run.native, packed);
  const double query_encrypt_import_ms = elapsed(start);
  const auto before = run.runtime.stats().live_bytes;
  start = Clock::now();
  Value preparation;
  if (run.prepared)
    preparation = run.runtime.submit({"contrib.search.prepare-query", {encrypted_query}, uint32_t{width}}).at(0);
  const double prepare_ms = elapsed(start);
  const auto prep_bytes = run.runtime.stats().live_bytes - before;
  const Values arguments =
    run.prepared ? Values{encrypted_query, run.index, preparation} : Values{encrypted_query, run.index};
  auto evaluate = [&] {
    if (!run.session)
      return run.runtime.submit({"contrib.search.score", arguments, {}});
    return run.session->run([&](Scope &scope) {
      ScopedSearch program(scope);
      std::optional<ScopeValue> state;
      if (run.prepared)
        state = scope.input(preparation);
      return program.score(EncryptedQuery{encrypted_query}, run.documents, state);
    });
  };
  const auto expected = oracle(fixture, query);
  constexpr double tolerance = 1e-5;
  double max_error = 0, decrypt_extract_ms = 0, verification_decrypt_total_ms = 0;
  std::vector<double> scores;
  auto verify = [&](const Values &results) {
    const auto decode_start = Clock::now();
    scores.clear();
    for (const auto &result : results) {
      auto decoded = client.decryptVector(run.native, result);
      for (unsigned lane = 0; lane < run.capacity && scores.size() < fixture.documents.size(); ++lane)
        scores.push_back(decoded[lane * width]);
    }
    decrypt_extract_ms = elapsed(decode_start);
    verification_decrypt_total_ms += decrypt_extract_ms;
    require(scores.size() == fixture.documents.size(), "wrong score count");
    for (size_t i = 0; i < scores.size(); ++i) {
      require(std::isfinite(scores[i]), "non-finite decrypted score");
      max_error = std::max(max_error, std::abs(scores[i] - expected[i]));
    }
    require(max_error <= tolerance, "encrypted scores disagree with plaintext oracle");
  };
  // Warm-up is reported separately. Every timed call starts with no live outputs.
  start = Clock::now();
  auto results = evaluate();
  const double warmup_ms = elapsed(start);
  verify(results);
  results.clear();
  std::vector<double> samples;
  for (unsigned i = 0; i < repetitions; ++i) {
    results.clear();
    start = Clock::now();
    results = evaluate();
    samples.push_back(elapsed(start));
    verify(results); // Every result is checked, outside its evaluation interval.
  }
  start = Clock::now();
  const auto order = rank(fixture, scores);
  const double client_rank_ms = elapsed(start);
  const auto plain_order = rank(fixture, expected);
  const size_t k = std::min<size_t>(5, order.size()), quality_k = std::min<size_t>(3, order.size());
  size_t overlap = 0, hits = 0, available = 0;
  bool exact = true, tolerance_consistent = true;
  std::set<std::string> relevant(query.relevant.begin(), query.relevant.end());
  for (const auto &doc : fixture.documents)
    available += relevant.count(doc.id);
  json top = json::array(), plain_top = json::array();
  for (size_t i = 0; i < k; ++i) {
    const auto d = order[i];
    exact = exact && d == plain_order[i];
    overlap += std::find(plain_order.begin(), plain_order.begin() + k, d) != plain_order.begin() + k;
    // A rank inversion is acceptable only within the documented numeric tolerance.
    tolerance_consistent = tolerance_consistent && std::abs(expected[d] - expected[plain_order[i]]) <= 2 * tolerance;
    top.push_back({{"id", fixture.documents[d].id},
                   {"title", fixture.documents[d].title},
                   {"source", fixture.documents[d].source},
                   {"text", fixture.documents[d].text},
                   {"score", scores[d]},
                   {"plaintext_score", expected[d]}});
    plain_top.push_back(fixture.documents[plain_order[i]].id);
  }
  for (size_t i = 0; i < quality_k; ++i)
    hits += relevant.count(fixture.documents[order[i]].id);
  require(tolerance_consistent, "top-k ordering differs beyond numeric tolerance");
  std::set<std::string> selected;
  for (const auto &event : run.runtime.trace())
    if (event.kind == TraceKind::Selected && event.operation.rfind("contrib.search", 0) == 0)
      selected.insert(event.recipe);
  size_t output_bytes = 0, output_reserved_bytes = 0;
  for (const auto &result : results) {
    output_bytes += payloadBytes(client.computeServices(), run.native, result);
    output_reserved_bytes += result.bytes();
  }
  const auto query_bytes = payloadBytes(client.computeServices(), run.native, encrypted_query);
  json record{{"query_id", query.id},
              {"mode", run.name},
              {"query_encrypt_import_ms", query_encrypt_import_ms},
              {"prepare_ms", prepare_ms},
              {"preparation_bytes", prep_bytes},
              {"warmup_ms", warmup_ms},
              {"evaluation_ms", samples},
              {"median_evaluation_ms", median(samples)},
              {"decrypt_extract_ms", decrypt_extract_ms},
              {"verification_decrypt_total_ms", verification_decrypt_total_ms},
              {"verified_evaluations", repetitions + 1},
              {"client_rank_ms", client_rank_ms},
              {"max_absolute_error", max_error},
              {"score_tolerance", tolerance},
              {"top_k", k},
              {"top_k_exact_order", exact},
              {"top_k_overlap", overlap},
              {"top_k_tolerance_consistent", tolerance_consistent},
              {"plaintext_top_ids", plain_top},
              {"label_hit_at_3", available ? json(hits > 0) : json(nullptr)},
              {"label_hits_at_3", available ? json(hits) : json(nullptr)},
              {"available_relevance_labels", available},
              {"label_recall_at_3", available ? json(double(hits) / available) : json(nullptr)},
              {"top_results", top},
              {"selected_search_recipes", selected},
              {"query_ciphertexts", 1},
              {"query_ciphertext_bytes", query_bytes},
              {"query_ciphertext_reserved_bytes", encrypted_query.bytes()},
              {"output_ciphertexts", results.size()},
              {"output_ciphertext_bytes", output_bytes},
              {"output_ciphertext_reserved_bytes", output_reserved_bytes},
              {"live_ciphertexts", run.index.dependencies().size() + 1 + results.size()},
              {"live_ciphertext_bytes", run.corpus_ciphertext_bytes + query_bytes + output_bytes},
              {"retained_bytes_with_outputs", run.runtime.stats().live_bytes},
              {"mode_lifetime_peak_tracked_bytes", run.runtime.stats().peak_live_bytes}};
  if (run.session) {
    record["scope_span_trace_scope"] = "bounded mode-lifetime snapshot; may include earlier queries/evaluations";
    record["selected_scope_spans"] = json::array();
    for (const auto &event : run.runtime.trace())
      if (event.kind == TraceKind::Selected && event.operation == "scope.span")
        record["selected_scope_spans"].push_back({{"kernel", event.recipe}, {"span", event.detail}});
  }
  record["plaintext_top_k_boundary_margin"] =
    k < order.size() ? json(expected[plain_order[k - 1]] - expected[plain_order[k]]) : json(nullptr);
  return record;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (helpRequested(argc, argv)) {
      std::cout << benchmark_usage;
      return 0;
    }
    const auto opts = options(argc, argv);
    std::ifstream input(opts.fixture);
    require(bool(input), "cannot open fixture: " + opts.fixture);
    json source;
    input >> source;
    auto fixture = parseFixture(source);
    if (opts.limit && opts.limit < fixture.documents.size())
      fixture.documents.resize(opts.limit);
    std::vector<Query> queries;
    if (opts.all)
      queries = fixture.queries;
    else if (opts.query.empty())
      queries.push_back(fixture.queries.front());
    else {
      for (const auto &q : fixture.queries)
        if (q.id == opts.query)
          queries.push_back(q);
      require(!queries.empty(), "unknown query id: " + opts.query);
    }
    Runtime client_inputs;
    TrustedClient client(client_inputs, opts.library);
    std::vector<std::unique_ptr<Run>> modes;
    if (opts.compare) {
      modes.push_back(std::make_unique<Run>(client, opts, fixture, 1, false));
      modes.push_back(std::make_unique<Run>(client, opts, fixture, 1, true));
      modes.push_back(std::make_unique<Run>(client, opts, fixture, tile_capacity, false));
    }
    modes.push_back(std::make_unique<Run>(client, opts, fixture, tile_capacity, true));
    json report{
      {"schema_version", 1},
      {"model", fixture.model},
      {"build_type", FHN_SEARCH_BUILD_TYPE},
      {"documents", fixture.documents.size()},
      {"logical_slots", slots},
      {"padded_width", width},
      {"repetitions", opts.repetitions},
      {"key_setup_ms", client.keySetupMs()},
      {"index_layouts", json::array()},
      {"searches", json::array()},
      {"scope", "single-owner in-process demo; client embeds, encrypts, decrypts and ranks; compute scores ciphertext"},
      {"timing_scope", "warm synchronous score submission incl. dispatch/allocation; excludes key/index/query setup, "
                       "preparation, warm-up, decrypt and ranking"},
      {"memory_scope",
       "runtime charged ciphertext/preparation slabs and contribution metadata; excludes keys, scratch, "
       "unaccounted runtime/allocator overhead, model and other runtimes; not wire bytes or RSS"},
      {"baseline_scope", "same 4096-slot representation; row baseline is not claimed optimal"}};
    if (opts.scoped) {
      report["scoped_execution"] = true;
      report["scope_kernels_enabled"] = opts.scope_kernels;
      report["timing_scope"] = "warm synchronous scope capture, selection and score execution incl. allocation; "
                               "excludes key/index/query setup, preparation, warm-up, decrypt and ranking";
    }
    report["ciphertext_scope"] =
      "direct corpus/query/output Values: live native payload and reserved capacity separately; "
      "embedded preparation snapshots are accounted in preparation_bytes";
    for (const auto &mode : modes)
      report["index_layouts"].push_back(
        {{"mode", mode->name},
         {"capacity", mode->capacity},
         {"ciphertexts", (fixture.documents.size() + mode->capacity - 1) / mode->capacity},
         {"pack_encrypt_import_index_ms", mode->index_setup_ms},
         {"corpus_ciphertext_bytes", mode->corpus_ciphertext_bytes},
         {"corpus_ciphertext_reserved_bytes", mode->corpus_ciphertext_reserved_bytes},
         {"corpus_retained_bytes", mode->corpus_bytes}});
    std::cout << "Encrypted personal-document search benchmark: " << fixture.documents.size() << " documents, "
              << dimensions << " dimensions (padded " << width << ")\n"
              << "Both queries and documents encrypted; top-k is decrypted and ranked by the owner.\n";
    for (size_t q = 0; q < queries.size(); ++q) {
      std::cout << "\nQuestion [" << queries[q].id << "]: " << queries[q].text << '\n';
      // Rotate mode order across queries to reduce a fixed-order timing bias.
      for (size_t m = 0; m < modes.size(); ++m) {
        auto &mode = *modes[(m + q) % modes.size()];
        auto result = search(client, mode, fixture, queries[q], opts.repetitions);
        std::cout << "  " << mode.name << ": " << std::fixed << std::setprecision(3)
                  << result["median_evaluation_ms"].get<double>() << " ms, max error " << std::scientific
                  << result["max_absolute_error"].get<double>() << ", top-5 overlap " << result["top_k_overlap"] << '/'
                  << result["top_k"] << '\n';
        for (const auto &name : result["selected_search_recipes"])
          std::cout << "    " << name.get<std::string>() << '\n';
        if (mode.name == "tile-prepared") {
          for (const auto &entry : result["top_results"]) {
            std::cout << "    " << std::fixed << std::setprecision(4) << entry["score"].get<double>() << "  "
                      << entry["id"].get<std::string>() << " — " << entry["title"].get<std::string>() << '\n'
                      << "      " << entry["source"].get<std::string>() << '\n';
            const auto text = entry["text"].get<std::string>();
            std::cout << "      " << text.substr(0, text.find('\n')) << '\n';
          }
          const auto &hit = result["label_hit_at_3"];
          std::cout << "  Hand-labeled relevance hit@3: "
                    << (hit.is_null()     ? "unjudged"
                        : hit.get<bool>() ? "yes"
                                          : "no")
                    << " (model quality, separate from CKKS fidelity)\n";
        }
        report["searches"].push_back(std::move(result));
      }
    }
    if (!opts.output.empty()) {
      std::ofstream output(opts.output);
      require(bool(output), "cannot write JSON output: " + opts.output);
      output << std::setw(2) << report << '\n';
      output.close();
      require(bool(output), "failed to finish JSON output");
    }
    std::cout << "\nVerified all document scores against the plaintext cosine oracle.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "private-search-benchmark: " << error.what() << '\n';
    return 1;
  }
}
