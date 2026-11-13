#include "corpus_backend.h"
#include "corpus_circuit.h"
#include "corpus_oracle.h"
#include "corpus_shapes.h"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fhenomenon::corpus;

namespace {
using RealSlots = std::vector<double>;
using Values = std::map<uint32_t, RealSlots>;

// An independent value interpreter makes argument order, constants, signed
// rotations and invented temporary IDs observable without requiring CKKS.
Values evaluateCircuit(const FhnCircuit &circuit, Values values) {
  for (uint32_t i = 0; i < circuit.num_nodes; ++i) {
    const auto &node = circuit.nodes[i];
    const auto &a = values.at(node.operands[0]);
    RealSlots result(a.size());
    for (size_t slot = 0; slot < a.size(); ++slot) {
      switch (node.op) {
      case FHN_CIRCUIT_ADD:
        result[slot] = a[slot] + values.at(node.operands[1]).at(slot);
        break;
      case FHN_CIRCUIT_SUBTRACT:
        result[slot] = a[slot] - values.at(node.operands[1]).at(slot);
        break;
      case FHN_CIRCUIT_MULTIPLY:
        result[slot] = a[slot] * values.at(node.operands[1]).at(slot);
        break;
      case FHN_CIRCUIT_SCALE:
        result[slot] = a[slot] * node.scalar;
        break;
      case FHN_CIRCUIT_ADD_CONSTANT:
        result[slot] = a[slot] + node.scalar;
        break;
      case FHN_CIRCUIT_NEGATE:
        result[slot] = -a[slot];
        break;
      case FHN_CIRCUIT_ROTATE: {
        const auto count = static_cast<int64_t>(a.size());
        const auto distance = ((node.rotation % count) + count) % count;
        result[slot] = a[(slot + static_cast<size_t>(distance)) % a.size()];
        break;
      }
      case FHN_CIRCUIT_CONJUGATE:
        result[slot] = a[slot]; // Corpus inputs are real.
        break;
      default:
        throw std::runtime_error("unknown semantic circuit operation");
      }
    }
    if (!values.emplace(node.result_id, std::move(result)).second)
      throw std::runtime_error("import introduced a duplicate value ID");
  }
  return values;
}
} // namespace

TEST(CorpusCircuit, AllThirteenShapesHaveExplicitSemanticImportOutcomes) {
  auto shapes = allShapes();
  ASSERT_EQ(shapes.size(), 13u);
  uint32_t imported = 0, unsupported = 0;
  for (const auto &shape : shapes) {
    SCOPED_TRACE(shape.name);
    std::string error;
    auto circuit = importCircuit(*shape.program, &error);
    if (shape.name == "psi-eq") {
      EXPECT_FALSE(circuit);
      EXPECT_FALSE(error.empty());
      ++unsupported;
      continue;
    }
    ASSERT_TRUE(circuit) << error;
    const auto source_results = evaluate(*shape.program, shape.inputs);
    ASSERT_TRUE(source_results);
    Values inputs;
    for (const auto &[id, slots] : shape.inputs)
      inputs.emplace(id, RealSlots(slots.begin(), slots.end()));
    const auto view = circuit->view();
    ASSERT_EQ(view.version, FHN_CIRCUIT_VERSION);
    ASSERT_EQ(view.num_inputs, shape.program->num_inputs);
    ASSERT_EQ(view.num_outputs, shape.output_ids.size());
    for (uint32_t i = 0; i < view.num_inputs; ++i)
      EXPECT_EQ(view.input_ids[i], shape.program->input_ids[i]);
    const auto actual = evaluateCircuit(view, std::move(inputs));
    for (uint32_t i = 0; i < view.num_outputs; ++i) {
      const auto id = shape.output_ids[i];
      EXPECT_EQ(view.output_ids[i], id);
      const auto &expected = source_results->at(id);
      EXPECT_EQ(actual.at(id), RealSlots(expected.begin(), expected.end()));
    }
    ++imported;
  }
  EXPECT_EQ(imported, 12u); // Depth is a compiler decision, not import failure.
  EXPECT_EQ(unsupported, 1u);
}

TEST(CorpusCircuit, ArithmeticPreservesFractionalConstantsAndOperandOrder) {
  FhnInstruction instructions[]{
    {FHN_ADD_CC, 20, {7, 100, 0, 0}, {}, {}},       {FHN_SUB_CC, 21, {7, 100, 0, 0}, {}, {}},
    {FHN_MULT_CC, 22, {7, 100, 0, 0}, {}, {}},      {FHN_HMULT, 23, {7, 100, 0, 0}, {}, {}},
    {FHN_MULT_CS, 24, {7, 0, 0, 0}, {}, {-0.5, 0}}, {FHN_ADD_CS, 25, {7, 0, 0, 0}, {}, {0.25, 0}},
    {FHN_NEGATE, 26, {7, 0, 0, 0}, {}, {}},         {FHN_ROTATE, 27, {7, 0, 0, 0}, {-1, 0, 0, 0}, {}},
    {FHN_CONJUGATE, 28, {7, 0, 0, 0}, {}, {}}};
  uint32_t inputs[]{7, 100};
  uint32_t outputs[]{20, 21, 22, 23, 24, 25, 26, 27, 28};
  FhnProgram program{FHN_ABI_VERSION, 9, instructions, 2, inputs, 9, outputs};
  std::string error;
  auto circuit = importCircuit(program, &error);
  ASSERT_TRUE(circuit) << error;
  const auto values = evaluateCircuit(circuit->view(), {{7, {1, -2, 3, 4}}, {100, {5, 6, -7, 8}}});
  EXPECT_EQ(values.at(20), (RealSlots{6, 4, -4, 12}));
  EXPECT_EQ(values.at(21), (RealSlots{-4, -8, 10, -4}));
  EXPECT_EQ(values.at(22), (RealSlots{5, -12, -21, 32}));
  EXPECT_EQ(values.at(23), (RealSlots{5, -12, -21, 32}));
  EXPECT_EQ(values.at(24), (RealSlots{-0.5, 1, -1.5, -2}));
  EXPECT_EQ(values.at(25), (RealSlots{1.25, -1.75, 3.25, 4.25}));
  EXPECT_EQ(values.at(26), (RealSlots{-1, 2, -3, -4}));
  EXPECT_EQ(values.at(27), (RealSlots{4, 1, -2, 3}));
  EXPECT_EQ(values.at(28), (RealSlots{1, -2, 3, 4}));
}

TEST(CorpusCircuit, RotateAddExpansionPreservesDistinctOperandsAndFutureIds) {
  // Sparse IDs and later references catch temporary collisions; distinct
  // inputs catch accidentally expanding rotate(a)+a instead of rotate(a)+b.
  FhnInstruction instructions[]{{FHN_HROT_ADD, 2, {1, 40, 0, 0}, {1, 0, 0, 0}, {}},
                                {FHN_HROT_ADD, 41, {2, 40, 0, 0}, {-1, 0, 0, 0}, {}},
                                {FHN_SUB_CC, 42, {41, 2, 0, 0}, {}, {}}};
  uint32_t inputs[]{1, 40};
  uint32_t outputs[]{2, 41, 42};
  FhnProgram program{FHN_ABI_VERSION, 3, instructions, 2, inputs, 3, outputs};
  std::string error;
  auto circuit = importCircuit(program, &error);
  ASSERT_TRUE(circuit) << error;
  const auto values = evaluateCircuit(circuit->view(), {{1, {1, -2, 3, 4}}, {40, {5, 6, -7, 8}}});
  EXPECT_EQ(values.at(2), (RealSlots{3, 9, -3, 9}));
  EXPECT_EQ(values.at(41), (RealSlots{14, 9, 2, 5}));
  EXPECT_EQ(values.at(42), (RealSlots{11, 0, 5, -4}));
}

TEST(CorpusCircuit, ImportedValuesSurviveSourceMutationDestructionAndMove) {
  auto moved = []() {
    FhnInstruction instructions[]{{FHN_ADD_CS, 10, {9, 0, 0, 0}, {}, {0.5, 0}}};
    uint32_t inputs[]{9}, outputs[]{10};
    FhnProgram source{FHN_ABI_VERSION, 1, instructions, 1, inputs, 1, outputs};
    auto imported = importCircuit(source);
    if (!imported)
      throw std::runtime_error("semantic import failed");
    instructions[0].fparams[0] = 1000;
    inputs[0] = 55;
    outputs[0] = 55;
    Circuit destination;
    destination = std::move(*imported);
    return destination;
  }();
  const auto view = moved.view();
  ASSERT_EQ(view.num_inputs, 1u);
  ASSERT_EQ(view.num_outputs, 1u);
  EXPECT_EQ(view.input_ids[0], 9u);
  EXPECT_EQ(view.output_ids[0], 10u);
  const auto result = evaluateCircuit(view, {{9, {-1.25, 2.5}}});
  EXPECT_EQ(result.at(10), (RealSlots{-0.75, 3}));
}

TEST(CorpusCircuit, PhysicalAndExactBooleanOperationsFailWithDiagnostics) {
  const FhnOpCode unsupported[]{FHN_RELINEARIZE, FHN_RESCALE, FHN_LEVEL_DOWN, FHN_MOD_DOWN, FHN_MULT_KEY, FHN_AND,
                                FHN_OR,          FHN_XOR,     FHN_EQ,         FHN_LT,       FHN_LE};
  uint32_t inputs[]{1, 2}, outputs[]{3};
  for (const auto op : unsupported) {
    SCOPED_TRACE(static_cast<int>(op));
    FhnInstruction instruction{op, 3, {1, 2, 0, 0}, {}, {}};
    FhnProgram program{FHN_ABI_VERSION, 1, &instruction, 2, inputs, 1, outputs};
    std::string error;
    EXPECT_FALSE(importCircuit(program, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(importCircuit(program)); // Diagnostics are optional.
  }
}

TEST(CorpusCircuit, CompositeMappingsPreserveValuesWithMaximumSparseInputId) {
  FhnInstruction instructions[]{{FHN_HROT, 5, {UINT32_MAX, 0, 0, 0}, {5, 0, 0, 0}, {}},
                                {FHN_HCONJ_ADD, 6, {UINT32_MAX, 2, 0, 0}, {}, {}},
                                {FHN_MAD, 7, {UINT32_MAX, 2, 0, 0}, {}, {0.25, 0}},
                                {FHN_SUB_SC, 8, {UINT32_MAX, 0, 0, 0}, {}, {0.5, 0}},
                                {FHN_HROT_ADD, 9, {UINT32_MAX, 2, 0, 0}, {-5, 0, 0, 0}, {}}};
  uint32_t inputs[]{UINT32_MAX, 2}, outputs[]{5, 6, 7, 8, 9};
  FhnProgram program{FHN_ABI_VERSION, 5, instructions, 2, inputs, 5, outputs};
  std::string error;
  auto circuit = importCircuit(program, &error);
  ASSERT_TRUE(circuit) << error;
  const auto values = evaluateCircuit(circuit->view(), {{UINT32_MAX, {1, -2, 3, 4}}, {2, {5, 6, -7, 8}}});
  EXPECT_EQ(values.at(5), (RealSlots{-2, 3, 4, 1}));
  EXPECT_EQ(values.at(6), (RealSlots{6, 4, -4, 12}));
  EXPECT_EQ(values.at(7), (RealSlots{5.25, 5.5, -6.25, 9}));
  EXPECT_EQ(values.at(8), (RealSlots{-0.5, 2.5, -2.5, -3.5}));
  EXPECT_EQ(values.at(9), (RealSlots{9, 7, -9, 11}));
}

TEST(CorpusCircuit, InvalidVersionsAndMissingDeclaredArraysFailWithDiagnostics) {
  const FhnProgram invalid[]{{FHN_ABI_VERSION + 1, 0, nullptr, 0, nullptr, 0, nullptr},
                             {FHN_ABI_VERSION, 1, nullptr, 0, nullptr, 0, nullptr},
                             {FHN_ABI_VERSION, 0, nullptr, 1, nullptr, 0, nullptr},
                             {FHN_ABI_VERSION, 0, nullptr, 0, nullptr, 1, nullptr}};
  for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
    SCOPED_TRACE(i);
    std::string error;
    EXPECT_FALSE(importCircuit(invalid[i], &error));
    EXPECT_FALSE(error.empty());
  }
}

TEST(CorpusCircuitBackend, LegacyBackendRemainsLoadableWithoutCircuitExports) {
  std::string error;
  auto backend = CorpusBackend::load(TOY_TEST_LIBRARY, "toyfhe_", &error);
  ASSERT_TRUE(backend) << error;
  EXPECT_FALSE(backend->hasCircuit());
  EXPECT_EQ(backend->runExecutable(), nullptr);
  EXPECT_EQ(backend->executableStats(), nullptr);
  EXPECT_EQ(backend->executableFree(), nullptr);
  char diagnostic[128]{};
  EXPECT_EQ(backend->compileCircuit(nullptr, nullptr, 0, nullptr, diagnostic, sizeof(diagnostic)), nullptr);
  EXPECT_NE(backend->prepareExecutable(nullptr), 0);
}

TEST(CorpusCircuitBackend, NativeCircuitRunsAfterLoaderMoveAndAcceptsFreshEncryptedInputs) {
  std::string error;
  auto original = CorpusBackend::load(CKKS_TEST_LIBRARY, "ckks_", &error);
  ASSERT_TRUE(original) << error;
  CorpusBackend moved(std::move(*original));
  auto replacement = CorpusBackend::load(CKKS_TEST_LIBRARY, "ckks_", &error);
  ASSERT_TRUE(replacement) << error;
  *replacement = std::move(moved);
  auto &backend = *replacement;
  ASSERT_TRUE(backend.hasCircuit());
  ASSERT_NE(backend.runExecutable(), nullptr);
  ASSERT_NE(backend.executableStats(), nullptr);
  ASSERT_NE(backend.executableFree(), nullptr);

  auto release = [ctx = backend.ctx(), free = backend.bufferFree()](FhnBuffer *buffer) { free(ctx, buffer); };
  using Buffer = std::unique_ptr<FhnBuffer, decltype(release)>;
  Buffer input(backend.bufferAlloc()(backend.ctx()), release);
  Buffer output(backend.bufferAlloc()(backend.ctx()), release);
  ASSERT_TRUE(input);
  ASSERT_TRUE(output);
  const double initial[]{0.125, -1.5, 2.25, 0.75};
  ASSERT_EQ(backend.encryptPacked(input.get(), initial, 4), 0);
  FhnCircuitNode nodes[]{{FHN_CIRCUIT_MULTIPLY, 2, {1, 1}, 0, 0},
                         {FHN_CIRCUIT_ROTATE, 3, {2, 0}, 1, 0},
                         {FHN_CIRCUIT_ADD_CONSTANT, 4, {3, 0}, 0, 0.5}};
  const uint32_t input_ids[]{1}, output_ids[]{4};
  const FhnCircuit circuit{FHN_CIRCUIT_VERSION, 3, nodes, 1, input_ids, 1, output_ids};
  const FhnBuffer *inputs[]{input.get()};
  FhnBuffer *outputs[]{output.get()};
  char diagnostic[256]{};
  std::unique_ptr<FhnExecutable, FhnExecutableFreeFn> executable(
    backend.compileCircuit(&circuit, inputs, 1, nullptr, diagnostic, sizeof(diagnostic)), backend.executableFree());
  ASSERT_TRUE(executable) << diagnostic;
  ASSERT_EQ(backend.prepareExecutable(executable.get()), 0);
  FhnExecutableStats stats{};
  stats.struct_size = sizeof(stats);
  ASSERT_EQ(backend.executableStats()(executable.get(), &stats), 0);
  EXPECT_GT(stats.regions, 0u);
  ASSERT_EQ(backend.runExecutable()(executable.get(), inputs, 1, outputs, 1), 0);
  double actual[4]{};
  ASSERT_EQ(backend.decryptPacked(output.get(), actual, 4), 0);
  const double expected[]{2.75, 5.5625, 1.0625, 0.515625};
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-4) << "slot " << i;

  // Compiled execution must bind new ciphertext contents rather than retain
  // values from compilation or the first run.
  const double fresh[]{-2, 0.25, -0.5, 1.5};
  ASSERT_EQ(backend.encryptPacked(input.get(), fresh, 4), 0);
  ASSERT_EQ(backend.runExecutable()(executable.get(), inputs, 1, outputs, 1), 0);
  ASSERT_EQ(backend.decryptPacked(output.get(), actual, 4), 0);
  const double fresh_expected[]{0.5625, 0.75, 2.75, 4.5};
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(actual[i], fresh_expected[i], 1e-4) << "fresh slot " << i;
}
