#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cfenv>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

class Sha256 {
public:
    void update(const void* source, std::size_t length) {
        const auto* bytes = static_cast<const std::uint8_t*>(source);
        total_ += length;
        while (length) {
            const auto count = std::min(length, block_.size() - used_);
            std::copy_n(bytes, count, block_.begin() + static_cast<std::ptrdiff_t>(used_));
            used_ += count; bytes += count; length -= count;
            if (used_ == block_.size()) { transform(); used_ = 0; }
        }
    }
    std::string finish() {
        const auto bits = static_cast<std::uint64_t>(total_) * 8U;
        const std::uint8_t marker = 0x80; update(&marker, 1);
        const std::uint8_t zero = 0;
        while (used_ != 56) update(&zero, 1);
        std::array<std::uint8_t, 8> length{};
        for (int index = 0; index < 8; ++index) length[7 - index] = static_cast<std::uint8_t>(bits >> (index * 8));
        update(length.data(), length.size());
        std::ostringstream output; output << std::hex << std::setfill('0');
        for (auto value : state_) output << std::setw(8) << value;
        return output.str();
    }
private:
    static std::uint32_t rotate(std::uint32_t value, unsigned count) { return (value >> count) | (value << (32U - count)); }
    void transform() {
        static constexpr std::array<std::uint32_t, 64> constants{
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) words[i] = (static_cast<std::uint32_t>(block_[4*i]) << 24U) | (static_cast<std::uint32_t>(block_[4*i+1]) << 16U) | (static_cast<std::uint32_t>(block_[4*i+2]) << 8U) | block_[4*i+3];
        for (std::size_t i = 16; i < 64; ++i) { const auto s0=rotate(words[i-15],7)^rotate(words[i-15],18)^(words[i-15]>>3U); const auto s1=rotate(words[i-2],17)^rotate(words[i-2],19)^(words[i-2]>>10U); words[i]=words[i-16]+s0+words[i-7]+s1; }
        auto [a,b,c,d,e,f,g,h] = state_;
        for (std::size_t i=0;i<64;++i) { const auto s1=rotate(e,6)^rotate(e,11)^rotate(e,25); const auto choice=(e&f)^((~e)&g); const auto t1=h+s1+choice+constants[i]+words[i]; const auto s0=rotate(a,2)^rotate(a,13)^rotate(a,22); const auto majority=(a&b)^(a&c)^(b&c); const auto t2=s0+majority; h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }
    std::array<std::uint32_t,8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<std::uint8_t,64> block_{}; std::size_t used_{}; std::size_t total_{};
};

std::string sha256(const std::string& value) { Sha256 hash; hash.update(value.data(), value.size()); return hash.finish(); }
std::string sha256_file(const std::string& path) { std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("cannot open checkpoint"); Sha256 hash; std::array<char,65536> buffer{}; while (input) { input.read(buffer.data(), buffer.size()); hash.update(buffer.data(), static_cast<std::size_t>(input.gcount())); } return hash.finish(); }

std::vector<std::string> split(const std::string& value, char separator) {
    if (value == "-") return {};
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, separator)) parts.push_back(part);
    return parts;
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    // Preserve infinities and signed zero. NaNs retain their upper payload bits and
    // are explicitly quieted so truncation/rounding can never turn them into zero.
    if ((bits & 0x7f800000U) == 0x7f800000U) {
        auto upper = static_cast<std::uint16_t>(bits >> 16U);
        if ((bits & 0x007fffffU) != 0) upper = static_cast<std::uint16_t>(upper | 0x0040U);
        return upper;
    }
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

enum class DType { BF16, F32 };

struct Tensor {
    DType dtype;
    std::string checkpoint_name;
    std::size_t offset{};
    std::size_t length{};
    std::vector<std::size_t> shape;
    const std::byte* data{};
};

struct GraphValue {
    DType dtype;
    std::vector<std::string> symbolic_shape;
    std::vector<std::size_t> shape;
    std::string lifetime;
};

struct Op {
    std::string kind;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> tensors;
    std::unordered_map<std::string, std::string> attributes;
};

struct Plan {
    std::string architecture_id;
    std::string config_identity;
    std::string weight_dtype;
    std::string activation_dtype;
    std::string output_dtype;
    std::string accumulator_dtype;
    std::string accumulation_order;
    std::string rounding;
    std::string contraction_policy;
    std::string softmax_policy;
    std::string transcendental_policy;
    std::string gelu_formula;
    std::string silu_formula;
    std::string rope_formula;
    std::string plan_sha256;
    std::string checkpoint_sha256;
    int eos_token_id{};
    bool has_eos{};
    std::unordered_map<std::string, std::size_t> shapes;
    std::unordered_map<std::string, Tensor> tensors;
    std::unordered_map<std::string, GraphValue> values;
    std::vector<Op> ops;
    bool canonical_serialization{};
    bool canonical_digest{};
    std::string checkpoint_path;
};

bool valid_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char item) { return (item >= '0' && item <= '9') || (item >= 'a' && item <= 'f'); });
}

std::size_t parse_size(const std::string& value) {
    if (value.empty() || (value.size() > 1 && value[0] == '0')) throw std::runtime_error("malformed numeric value");
    std::size_t parsed{};
    for (const unsigned char item : value) {
        if (item < '0' || item > '9') throw std::runtime_error("malformed numeric value");
        const auto digit = static_cast<std::size_t>(item - '0');
        if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10U) throw std::runtime_error("numeric value out of range");
        parsed = parsed * 10U + digit;
    }
    return parsed;
}

int parse_int(const std::string& value) {
    if (value.empty() || value == "-0" || (value[0] == '0' && value.size() > 1) || (value[0] == '-' && (value.size() == 1 || (value.size() > 2 && value[1] == '0')))) throw std::runtime_error("malformed numeric value");
    std::size_t used{}; long long parsed{}; try { parsed = std::stoll(value, &used); } catch (...) { throw std::runtime_error("malformed numeric value"); }
    if (used != value.size() || parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) throw std::runtime_error("malformed numeric value");
    return static_cast<int>(parsed);
}

DType parse_dtype(const std::string& value) {
    if (value == "F32") return DType::F32;
    if (value == "BF16") return DType::BF16;
    throw std::runtime_error("unsupported dtype " + value);
}

std::string dtype_name(DType value) {
    return value == DType::F32 ? "F32" : "BF16";
}

struct JsonValue {
    enum class Kind { Object, Array, String, Number, Boolean, Null } kind;
    std::map<std::string, JsonValue> object;
    std::vector<JsonValue> array;
    std::string text;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& source) : source_(source) {}

    JsonValue parse() {
        auto value = parse_value();
        whitespace();
        if (position_ != source_.size()) fail();
        return value;
    }

private:
    [[noreturn]] static void fail() { throw std::runtime_error("malformed safetensors header JSON"); }
    void whitespace() {
        while (position_ < source_.size() && (source_[position_] == ' ' || source_[position_] == '\n' || source_[position_] == '\r' || source_[position_] == '\t')) ++position_;
    }
    bool consume(char expected) {
        whitespace();
        if (position_ < source_.size() && source_[position_] == expected) { ++position_; return true; }
        return false;
    }
    void literal(const char* value) {
        while (*value) { if (position_ >= source_.size() || source_[position_++] != *value++) fail(); }
    }
    static int hex(char value) {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        fail();
    }
    std::uint32_t unicode_escape() {
        if (position_ + 4 > source_.size()) fail();
        std::uint32_t result = 0;
        for (int index = 0; index < 4; ++index) result = (result << 4U) | static_cast<std::uint32_t>(hex(source_[position_++]));
        return result;
    }
    static void append_utf8(std::string& output, std::uint32_t value) {
        if (value <= 0x7fU) output.push_back(static_cast<char>(value));
        else if (value <= 0x7ffU) { output.push_back(static_cast<char>(0xc0U | (value >> 6U))); output.push_back(static_cast<char>(0x80U | (value & 0x3fU))); }
        else if (value <= 0xffffU) { output.push_back(static_cast<char>(0xe0U | (value >> 12U))); output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU))); output.push_back(static_cast<char>(0x80U | (value & 0x3fU))); }
        else { output.push_back(static_cast<char>(0xf0U | (value >> 18U))); output.push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3fU))); output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU))); output.push_back(static_cast<char>(0x80U | (value & 0x3fU))); }
    }
    std::string parse_string() {
        whitespace();
        if (position_ >= source_.size() || source_[position_++] != '"') fail();
        std::string result;
        while (position_ < source_.size()) {
            const auto item = static_cast<unsigned char>(source_[position_++]);
            if (item == '"') return result;
            if (item < 0x20U) fail();
            if (item != '\\') { result.push_back(static_cast<char>(item)); continue; }
            if (position_ >= source_.size()) fail();
            switch (source_[position_++]) {
                case '"': result.push_back('"'); break; case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break; case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break; case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break; case 't': result.push_back('\t'); break;
                case 'u': {
                    auto code = unicode_escape();
                    if (code >= 0xd800U && code <= 0xdbffU) {
                        if (position_ + 2 > source_.size() || source_[position_++] != '\\' || source_[position_++] != 'u') fail();
                        const auto low = unicode_escape();
                        if (low < 0xdc00U || low > 0xdfffU) fail();
                        code = 0x10000U + ((code - 0xd800U) << 10U) + (low - 0xdc00U);
                    } else if (code >= 0xdc00U && code <= 0xdfffU) fail();
                    append_utf8(result, code); break;
                }
                default: fail();
            }
        }
        fail();
    }
    JsonValue parse_number() {
        whitespace(); const auto start = position_;
        if (position_ < source_.size() && source_[position_] == '-') ++position_;
        if (position_ >= source_.size()) fail();
        if (source_[position_] == '0') ++position_;
        else { if (source_[position_] < '1' || source_[position_] > '9') fail(); while (position_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_; }
        if (position_ < source_.size() && source_[position_] == '.') { ++position_; if (position_ >= source_.size() || !std::isdigit(static_cast<unsigned char>(source_[position_]))) fail(); while (position_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_; }
        if (position_ < source_.size() && (source_[position_] == 'e' || source_[position_] == 'E')) { ++position_; if (position_ < source_.size() && (source_[position_] == '+' || source_[position_] == '-')) ++position_; if (position_ >= source_.size() || !std::isdigit(static_cast<unsigned char>(source_[position_]))) fail(); while (position_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_; }
        return JsonValue{JsonValue::Kind::Number, {}, {}, source_.substr(start, position_ - start)};
    }
    JsonValue parse_value() {
        whitespace(); if (position_ >= source_.size()) fail();
        if (source_[position_] == '{') return parse_object();
        if (source_[position_] == '[') return parse_array();
        if (source_[position_] == '"') return JsonValue{JsonValue::Kind::String, {}, {}, parse_string()};
        if (source_[position_] == 't') { literal("true"); return JsonValue{JsonValue::Kind::Boolean}; }
        if (source_[position_] == 'f') { literal("false"); return JsonValue{JsonValue::Kind::Boolean}; }
        if (source_[position_] == 'n') { literal("null"); return JsonValue{JsonValue::Kind::Null}; }
        return parse_number();
    }
    JsonValue parse_object() {
        if (!consume('{')) fail(); JsonValue result{JsonValue::Kind::Object};
        if (consume('}')) return result;
        while (true) {
            auto key = parse_string(); if (!consume(':')) fail(); auto value = parse_value();
            if (!result.object.emplace(std::move(key), std::move(value)).second) throw std::runtime_error("duplicate safetensors header key");
            if (consume('}')) return result; if (!consume(',')) fail();
        }
    }
    JsonValue parse_array() {
        if (!consume('[')) fail(); JsonValue result{JsonValue::Kind::Array};
        if (consume(']')) return result;
        while (true) { result.array.push_back(parse_value()); if (consume(']')) return result; if (!consume(',')) fail(); }
    }
    const std::string& source_; std::size_t position_{};
};

struct CheckpointTensor { DType dtype; std::vector<std::size_t> shape; std::size_t offset; std::size_t length; };

std::size_t json_size(const JsonValue& value) {
    if (value.kind != JsonValue::Kind::Number || value.text.empty() || value.text[0] == '-' || value.text.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("malformed safetensors tensor integer");
    return parse_size(value.text);
}

std::unordered_map<std::string, CheckpointTensor> read_checkpoint_tensors(const std::string& path) {
    std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("cannot open checkpoint");
    input.seekg(0, std::ios::end); const auto end_position = input.tellg();
    if (end_position < 8) throw std::runtime_error("truncated safetensors header");
    const auto file_size = static_cast<std::uint64_t>(end_position); input.seekg(0);
    std::array<unsigned char, 8> prefix{}; input.read(reinterpret_cast<char*>(prefix.data()), 8);
    std::uint64_t header_length = 0; for (int index = 7; index >= 0; --index) header_length = (header_length << 8U) | prefix[static_cast<std::size_t>(index)];
    if (header_length > file_size - 8 || header_length > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("invalid safetensors header length");
    std::string header(static_cast<std::size_t>(header_length), '\0'); input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (static_cast<std::size_t>(input.gcount()) != header.size()) throw std::runtime_error("truncated safetensors header");
    const auto root = JsonParser(header).parse(); if (root.kind != JsonValue::Kind::Object) throw std::runtime_error("safetensors header must be an object");
    const auto data_start_u64 = 8U + header_length; if (data_start_u64 > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("safetensors offset overflow");
    const auto data_start = static_cast<std::size_t>(data_start_u64); const auto data_bytes = static_cast<std::size_t>(file_size - data_start_u64);
    std::unordered_map<std::string, CheckpointTensor> tensors;
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    for (const auto& [name, entry] : root.object) {
        if (name == "__metadata__") continue;
        if (entry.kind != JsonValue::Kind::Object || entry.object.size() != 3 || !entry.object.contains("dtype") || !entry.object.contains("shape") || !entry.object.contains("data_offsets")) throw std::runtime_error("malformed safetensors tensor metadata");
        const auto& dtype = entry.object.at("dtype"); const auto& shape = entry.object.at("shape"); const auto& offsets = entry.object.at("data_offsets");
        if (dtype.kind != JsonValue::Kind::String || shape.kind != JsonValue::Kind::Array || offsets.kind != JsonValue::Kind::Array || offsets.array.size() != 2) throw std::runtime_error("malformed safetensors tensor metadata");
        CheckpointTensor tensor{parse_dtype(dtype.text), {}, 0, 0};
        if (shape.array.empty()) throw std::runtime_error("empty safetensors tensor shape");
        std::size_t elements = 1; for (const auto& dimension_value : shape.array) { const auto dimension = json_size(dimension_value); if (!dimension) throw std::runtime_error("zero safetensors tensor dimension"); if (elements > std::numeric_limits<std::size_t>::max() / dimension) throw std::runtime_error("safetensors tensor shape overflow"); elements *= dimension; tensor.shape.push_back(dimension); }
        const auto start = json_size(offsets.array[0]); const auto finish = json_size(offsets.array[1]); if (finish < start || finish > data_bytes) throw std::runtime_error("safetensors tensor range outside checkpoint");
        const std::size_t element_size = tensor.dtype == DType::F32 ? 4 : 2; if (elements > std::numeric_limits<std::size_t>::max() / element_size || finish - start != elements * element_size) throw std::runtime_error("safetensors tensor byte length mismatch");
        if (start > std::numeric_limits<std::size_t>::max() - data_start) throw std::runtime_error("safetensors offset overflow");
        tensor.offset = data_start + start; tensor.length = finish - start; ranges.emplace_back(start, finish); tensors.emplace(name, std::move(tensor));
    }
    std::sort(ranges.begin(), ranges.end()); for (std::size_t index = 1; index < ranges.size(); ++index) if (ranges[index].first < ranges[index - 1].second) throw std::runtime_error("overlapping safetensors tensor ranges");
    return tensors;
}

std::string join(const std::vector<std::string>& values, char separator) {
    if (values.empty()) return "-"; std::string result;
    for (const auto& value : values) { if (!result.empty()) result += separator; result += value; }
    return result;
}

std::string canonical_plan(const Plan& plan) {
    std::string result;
    auto line = [&](const std::string& value) { result += value + '\n'; };
    line("identity\tarchitecture\t" + plan.architecture_id); line("identity\tconfig\t" + plan.config_identity); line("generation\teos_token_id\t" + std::to_string(plan.eos_token_id));
    std::map<std::string, std::size_t> shapes(plan.shapes.begin(), plan.shapes.end()); for (const auto& [name, value] : shapes) line("shape\t" + name + "\t" + std::to_string(value));
    line("semantic\tweight_dtype\t" + plan.weight_dtype); line("semantic\tactivation_dtype\t" + plan.activation_dtype); line("semantic\toutput_dtype\t" + plan.output_dtype); line("semantic\taccumulator_dtype\t" + plan.accumulator_dtype); line("semantic\taccumulation_order\t" + plan.accumulation_order); line("semantic\trounding\t" + plan.rounding); line("semantic\tcontraction_policy\t" + plan.contraction_policy); line("semantic\tsoftmax_policy\t" + plan.softmax_policy); line("semantic\ttranscendental_policy\t" + plan.transcendental_policy); line("semantic\tgelu_formula\t" + plan.gelu_formula); line("semantic\tsilu_formula\t" + plan.silu_formula); line("semantic\trope_formula\t" + plan.rope_formula);
    std::map<std::string, Tensor> tensors(plan.tensors.begin(), plan.tensors.end());
    for (const auto& [alias, tensor] : tensors) { std::vector<std::string> dimensions; for (auto value : tensor.shape) dimensions.push_back(std::to_string(value)); line("binding\t" + alias + "\t" + tensor.checkpoint_name + "\t" + dtype_name(tensor.dtype) + "\t" + join(dimensions, ',')); }
    std::map<std::string, GraphValue> values(plan.values.begin(), plan.values.end()); for (const auto& [name, value] : values) line("value\t" + name + "\t" + dtype_name(value.dtype) + "\t" + join(value.symbolic_shape, ',') + "\t" + value.lifetime);
    for (const auto& op : plan.ops) { std::map<std::string, std::string> attributes(op.attributes.begin(), op.attributes.end()); std::vector<std::string> encoded; for (const auto& [name, value] : attributes) encoded.push_back(name + "=" + value); line("op\t" + op.kind + "\t" + join(op.inputs, ',') + "\t" + join(op.outputs, ',') + "\t" + join(op.tensors, ',') + "\t" + join(encoded, ',')); }
    return result;
}

void validate_checkpoint_bindings(const Plan& plan, const std::string& checkpoint) {
    const auto checkpoint_tensors = read_checkpoint_tensors(checkpoint); std::unordered_set<std::string> bound;
    for (const auto& [alias, binding] : plan.tensors) {
        const auto found = checkpoint_tensors.find(binding.checkpoint_name);
        if (found == checkpoint_tensors.end() || !bound.insert(binding.checkpoint_name).second || found->second.dtype != binding.dtype || found->second.shape != binding.shape || found->second.offset != binding.offset || found->second.length != binding.length) throw std::runtime_error("checkpoint tensor binding mismatch for " + alias);
    }
    if (bound.size() != checkpoint_tensors.size()) throw std::runtime_error("unused checkpoint tensor binding");
}

Plan read_plan(const std::string& path, const std::string& checkpoint) {
    std::ifstream source(path, std::ios::binary);
    if (!source) throw std::runtime_error("cannot open execution plan");
    const std::string artifact((std::istreambuf_iterator<char>(source)), {});
    std::vector<std::string> lines; std::size_t start = 0;
    while (start < artifact.size()) { const auto end = artifact.find('\n', start); if (end == std::string::npos) throw std::runtime_error("execution plan must end with newline"); lines.push_back(artifact.substr(start, end-start)); start=end+1; }
    if (lines.size() < 5 || lines[0] != "STRPOT_EXECUTION_PLAN_V2") throw std::runtime_error("unsupported execution plan version");
    auto artifact_field = [&](std::size_t index, const std::string& name) { auto fields=split(lines[index], '\t'); if (fields.size()!=3 || fields[0]!="artifact" || fields[1]!=name || !valid_sha256(fields[2])) throw std::runtime_error("invalid artifact identity"); return fields[2]; };
    Plan plan; plan.plan_sha256=artifact_field(1,"plan_sha256"); plan.checkpoint_sha256=artifact_field(2,"checkpoint_sha256"); const auto body_digest=artifact_field(3,"body_sha256");
    const auto body_start = artifact.find('\n', artifact.find('\n', artifact.find('\n', artifact.find('\n')+1)+1)+1)+1;
    const auto body = artifact.substr(body_start);
    if (sha256(body) != body_digest) throw std::runtime_error("artifact integrity digest mismatch");
    if (sha256_file(checkpoint) != plan.checkpoint_sha256) throw std::runtime_error("checkpoint SHA-256 mismatch");
    std::string plan_bytes; bool data_started = false;
    std::unordered_set<std::string> identities, semantics, generations, data_aliases;
    for (std::size_t index=4; index<lines.size(); ++index) {
        const auto& line=lines[index]; if (line.empty()) throw std::runtime_error("empty execution plan record");
        auto fields=split(line,'\t');
        if (fields.size()==3 && fields[0]=="identity") {
            if (data_started || !identities.insert(fields[1]).second) throw std::runtime_error("duplicate plan identity");
            if (fields[1]=="architecture") plan.architecture_id=fields[2]; else if(fields[1]=="config") plan.config_identity=fields[2]; else throw std::runtime_error("unknown plan identity field"); plan_bytes += line+'\n';
        } else if(fields.size()==3 && fields[0]=="generation") {
            if(data_started || fields[1]!="eos_token_id" || !generations.insert(fields[1]).second) throw std::runtime_error("duplicate or unknown generation field"); plan.eos_token_id=parse_int(fields[2]); plan.has_eos=true; plan_bytes += line+'\n';
        } else if(fields.size()==3 && fields[0]=="shape") {
            if(data_started || !plan.shapes.emplace(fields[1],parse_size(fields[2])).second) throw std::runtime_error("duplicate plan shape"); plan_bytes += line+'\n';
        } else if(fields.size()==3 && fields[0]=="semantic") {
            if(data_started || !semantics.insert(fields[1]).second) throw std::runtime_error("duplicate numerical semantic");
            if(fields[1]=="weight_dtype") plan.weight_dtype=fields[2]; else if(fields[1]=="activation_dtype") plan.activation_dtype=fields[2]; else if(fields[1]=="output_dtype") plan.output_dtype=fields[2]; else if(fields[1]=="accumulator_dtype") plan.accumulator_dtype=fields[2]; else if(fields[1]=="accumulation_order") plan.accumulation_order=fields[2]; else if(fields[1]=="rounding") plan.rounding=fields[2]; else if(fields[1]=="contraction_policy") plan.contraction_policy=fields[2]; else if(fields[1]=="softmax_policy") plan.softmax_policy=fields[2]; else if(fields[1]=="transcendental_policy") plan.transcendental_policy=fields[2]; else if(fields[1]=="gelu_formula") plan.gelu_formula=fields[2]; else if(fields[1]=="silu_formula") plan.silu_formula=fields[2]; else if(fields[1]=="rope_formula") plan.rope_formula=fields[2]; else throw std::runtime_error("unknown numerical semantic"); plan_bytes += line+'\n';
        } else if(fields.size()==5 && fields[0]=="binding") {
            if(data_started) throw std::runtime_error("binding after tensor data"); std::vector<std::size_t> shape; for(const auto& item:split(fields[4],',')) { const auto dim=parse_size(item); if(!dim) throw std::runtime_error("zero tensor dimension"); shape.push_back(dim); }
            Tensor tensor{parse_dtype(fields[3]),fields[2],0,0,shape}; if(!plan.tensors.emplace(fields[1],std::move(tensor)).second) throw std::runtime_error("duplicate plan tensor alias "+fields[1]); plan_bytes += line+'\n';
        } else if(fields.size()==5 && fields[0]=="value") {
            if(data_started) throw std::runtime_error("value after tensor data"); GraphValue value{parse_dtype(fields[2]),split(fields[3],','),{},fields[4]}; if(!plan.values.emplace(fields[1],std::move(value)).second) throw std::runtime_error("duplicate graph value "+fields[1]); plan_bytes += line+'\n';
        } else if(fields.size()==6 && fields[0]=="op") {
            if(data_started) throw std::runtime_error("operator after tensor data"); Op op{fields[1],split(fields[2],','),split(fields[3],','),split(fields[4],','),{}}; for(const auto& item:split(fields[5],',')) { const auto equals=item.find('='); if(equals==std::string::npos || equals==0 || equals+1==item.size() || !op.attributes.emplace(item.substr(0,equals),item.substr(equals+1)).second) throw std::runtime_error("malformed or duplicate operator attribute"); } plan.ops.push_back(std::move(op)); plan_bytes += line+'\n';
        } else if(fields.size()==4 && fields[0]=="tensor_data") {
            data_started=true; auto found=plan.tensors.find(fields[1]); if(found==plan.tensors.end() || !data_aliases.insert(fields[1]).second) throw std::runtime_error("unknown or duplicate tensor data alias"); found->second.offset=parse_size(fields[2]); found->second.length=parse_size(fields[3]);
        } else throw std::runtime_error("invalid execution plan record");
    }
    const auto canonical = canonical_plan(plan);
    plan.canonical_serialization = plan_bytes == canonical;
    plan.canonical_digest = sha256(canonical) == plan.plan_sha256;
    plan.checkpoint_path = checkpoint;
    if(data_aliases.size()!=plan.tensors.size()) throw std::runtime_error("missing tensor data identity");
    return plan;
}

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open checkpoint");
        struct stat status {};
        if (fstat(fd_, &status) != 0) throw std::runtime_error("cannot stat checkpoint");
        size_ = static_cast<std::size_t>(status.st_size);
        data_ = static_cast<const std::byte*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) { data_ = nullptr; throw std::runtime_error("cannot map checkpoint"); }
    }
    ~MappedFile() {
        if (data_) munmap(const_cast<std::byte*>(data_), size_);
        if (fd_ >= 0) close(fd_);
    }
    const std::byte* at(std::size_t offset, std::size_t bytes) const {
        if (offset > size_ || bytes > size_ - offset) throw std::runtime_error("tensor outside checkpoint");
        return data_ + offset;
    }
private:
    int fd_{-1};
    std::size_t size_{};
    const std::byte* data_{};
};

// Reused from the portable native engine: persistent workers dynamically claim
// row ranges while the calling thread participates in the same dispatch.
constexpr std::size_t MAX_PARALLEL_THREADS = 256;

class Parallel {
public:
    static std::size_t checked_threads(std::size_t threads) {
        if (!threads || threads > MAX_PARALLEL_THREADS) throw std::runtime_error("threads exceed safe bound");
        return threads;
    }
    explicit Parallel(std::size_t threads) : completion_(checked_threads(threads)), threads_(threads) {
        const std::size_t worker_count = threads > 1 ? threads - 1 : 0;
        workers_.reserve(worker_count);
        try {
            for (std::size_t index = 0; index < worker_count; ++index) workers_.emplace_back([this] { worker_loop(); });
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
                ++generation_;
            }
            start_.notify_all();
            for (auto& worker : workers_) if (worker.joinable()) worker.join();
            throw;
        }
    }
    Parallel(const Parallel&) = delete;
    Parallel& operator=(const Parallel&) = delete;
    ~Parallel() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    template <typename Function>
    void run(std::size_t total, std::size_t grain, Function&& function) {
        ++dispatches_;
        if (workers_.empty() || total <= grain) {
            function(0, total);
            return;
        }
        ++threaded_dispatches_;
        {
            std::lock_guard lock(mutex_);
            function_ = std::forward<Function>(function);
            total_ = total;
            grain_ = grain;
            next_.store(0, std::memory_order_relaxed);
            ++generation_;
        }
        start_.notify_all();
        execute_chunks();
        completion_.arrive_and_wait();
        std::lock_guard lock(mutex_);
        function_ = nullptr;
    }
    std::size_t dispatches() const { return dispatches_; }
    std::size_t threaded_dispatches() const { return threaded_dispatches_; }
    std::size_t threads() const { return threads_; }
private:
    void execute_chunks() {
        while (true) {
            const std::size_t begin = next_.fetch_add(grain_, std::memory_order_relaxed);
            if (begin >= total_) return;
            function_(begin, std::min(begin + grain_, total_));
        }
    }
    void worker_loop() {
        std::size_t observed = 0;
        while (true) {
            {
                std::unique_lock lock(mutex_);
                start_.wait(lock, [this, observed] { return stopping_ || generation_ != observed; });
                if (stopping_) return;
                observed = generation_;
            }
            execute_chunks();
            completion_.arrive_and_wait();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::barrier<> completion_;
    bool stopping_{};
    std::size_t generation_{};
    std::size_t total_{};
    std::size_t grain_{1};
    std::atomic<std::size_t> next_{};
    std::function<void(std::size_t, std::size_t)> function_;
    std::size_t dispatches_{};
    std::size_t threaded_dispatches_{};
    std::size_t threads_{};
};

class Executor {
    struct Cache { std::vector<float> key; std::vector<float> value; std::size_t length{}; std::size_t kv_heads{}; std::size_t head_dim{}; };
    struct CompiledOp { const Op* op{}; std::vector<const Tensor*> tensors; };
public:
    Executor(Plan plan, const std::string& checkpoint, std::size_t threads)
        : plan_(std::move(plan)), weights_(checkpoint), parallel_(threads) {
        validate();
    }

    std::vector<int> generate(const std::vector<int>& prompt, std::size_t max_new) {
        if (prompt.empty()) throw std::runtime_error("prompt cannot be empty");
        if (!max_new) throw std::runtime_error("max_new must be positive");
        const auto context = shape("context_size");
        if (prompt.size() > context) throw std::runtime_error("generation exceeds plan context bound");
        const auto vocab = shape("vocab_size");
        for (const int token : prompt) {
            if (token < 0 || static_cast<std::size_t>(token) >= vocab) throw std::runtime_error("token outside embedding table");
        }
        const auto prefill_start = std::chrono::steady_clock::now();
        matrix_passes_ = 0;
        std::vector<float> logits = forward_block(prompt, position_);
        position_ += prompt.size();
        prefill_matrix_passes_ = matrix_passes_;
        prefill_physical_width_ = prompt.size();
        prefill_seconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prefill_start
        ).count();
        std::vector<int> result;
        for (std::size_t index = 0; index < max_new; ++index) {
            frontier_logits_.push_back(logits);
            int token = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
            result.push_back(token);
            if (token == plan_.eos_token_id || index + 1 == max_new) break;
            if (position_ >= context) throw std::runtime_error("generation exceeds plan context bound");
            const auto started = std::chrono::steady_clock::now();
            logits = forward(token, position_++);
            const auto finished = std::chrono::steady_clock::now();
            decode_intervals_.push_back(std::chrono::duration<double>(finished - started).count());
        }
        return result;
    }

    std::size_t executed() const { return executed_; }
    const std::vector<std::string>& trace() const { return trace_; }
    const std::vector<std::vector<float>>& frontier_logits() const { return frontier_logits_; }
    const std::vector<double>& decode_intervals() const { return decode_intervals_; }
    const Plan& plan() const { return plan_; }
    std::size_t matrix_passes() const { return matrix_passes_; }
    std::size_t parallel_dispatches() const { return parallel_.dispatches(); }
    std::size_t threaded_matrix_dispatches() const { return parallel_.threaded_dispatches(); }
    std::size_t threads() const { return parallel_.threads(); }
    std::size_t runtime_tensor_lookups() const { return runtime_tensor_lookups_; }
    std::size_t prefill_matrix_passes() const { return prefill_matrix_passes_; }
    double prefill_seconds() const { return prefill_seconds_; }
    std::size_t prefill_physical_width() const { return prefill_physical_width_; }
    const char* kernel_family() const { return "portable-tiled"; }
    std::vector<std::size_t> cache_lengths() const {
        std::vector<std::size_t> values;
        for (const auto& name : cache_order_) values.push_back(caches_.at(name).length);
        return values;
    }
    std::vector<std::vector<float>> cache_keys() const {
        std::vector<std::vector<float>> values;
        for (const auto& name : cache_order_) values.push_back(caches_.at(name).key);
        return values;
    }
    std::vector<std::vector<float>> cache_values() const {
        std::vector<std::vector<float>> values;
        for (const auto& name : cache_order_) values.push_back(caches_.at(name).value);
        return values;
    }

private:
    std::size_t shape(const std::string& name) const {
        auto found = plan_.shapes.find(name);
        if (found == plan_.shapes.end() || found->second == 0) throw std::runtime_error("missing plan shape " + name);
        return found->second;
    }
    const Tensor& tensor(const std::string& name) const {
        auto found = plan_.tensors.find(name);
        if (found == plan_.tensors.end()) throw std::runtime_error("unknown tensor alias " + name);
        return found->second;
    }
    static void arity(const Op& op, std::size_t inputs, std::size_t outputs, std::size_t tensors) {
        if (op.inputs.size() != inputs || op.outputs.size() != outputs || op.tensors.size() != tensors) {
            throw std::runtime_error("invalid " + op.kind + " operator arity");
        }
    }
    static void require_shape(const Tensor& tensor, const std::vector<std::size_t>& expected, const std::string& name) {
        if (tensor.shape != expected) throw std::runtime_error("tensor shape/orientation mismatch for " + name);
    }
    static void require_attributes(const Op& op, std::initializer_list<const char*> expected) {
        if (op.attributes.size() != expected.size()) throw std::runtime_error("invalid attributes for " + op.kind);
        for (const char* name : expected) if (!op.attributes.contains(name)) throw std::runtime_error("invalid attributes for " + op.kind);
    }
    std::size_t attr_size(const Op& op, const std::string& name) const {
        auto found = op.attributes.find(name);
        if (found == op.attributes.end()) throw std::runtime_error("operator " + op.kind + " missing attribute " + name);
        const auto parsed = parse_size(found->second);
        if (!parsed) throw std::runtime_error("invalid positive integer attribute " + name);
        return parsed;
    }
    float attr_float(const Op& op, const std::string& name) const {
        auto found = op.attributes.find(name);
        if (found == op.attributes.end()) throw std::runtime_error("operator " + op.kind + " missing attribute " + name);
        std::size_t used{}; float parsed{};
        try { parsed = std::stof(found->second, &used); } catch (...) { throw std::runtime_error("malformed numeric attribute " + name); }
        if (used != found->second.size()) throw std::runtime_error("malformed numeric attribute " + name);
        if (!std::isfinite(parsed) || parsed <= 0.0F) throw std::runtime_error("invalid positive finite attribute " + name);
        return parsed;
    }
    std::string attr(const Op& op, const std::string& name) const {
        auto found = op.attributes.find(name);
        if (found == op.attributes.end()) throw std::runtime_error("operator " + op.kind + " missing attribute " + name);
        return found->second;
    }

    void validate() {
        const std::size_t hidden = shape("hidden_size");
        const std::size_t vocab = shape("vocab_size");
        shape("context_size");
        if (plan_.shapes.size() != 3) throw std::runtime_error("plan requires exactly hidden_size, vocab_size, context_size");
        if (plan_.architecture_id.empty() || plan_.config_identity.empty()) throw std::runtime_error("missing plan identity");
        if (plan_.accumulator_dtype != "F32" || plan_.accumulation_order != "ordered" || (plan_.weight_dtype != "F32" && plan_.weight_dtype != "BF16") || (plan_.activation_dtype != "F32" && plan_.activation_dtype != "BF16") || (plan_.output_dtype != "F32" && plan_.output_dtype != "BF16") || plan_.rounding != "nearest_even" || plan_.contraction_policy != "disabled" || plan_.softmax_policy != "stable_max_subtraction" || plan_.transcendental_policy != "system_libm" || plan_.gelu_formula != "x_times_half_times_one_plus_erf_x_over_sqrt_two" || plan_.silu_formula != "x_over_one_plus_exp_neg_x" || plan_.rope_formula != "pair_rotation_theta_pow_2i_over_head_dim") {
            throw std::runtime_error("unsupported numerical semantics");
        }
        if (std::fegetround() != FE_TONEAREST) throw std::runtime_error("floating environment is not round-to-nearest-even");
        for (auto& [name, value] : plan_.tensors) {
            const bool f32 = value.dtype == DType::F32;
            if ((plan_.weight_dtype == "F32") != f32) throw std::runtime_error("mixed or undeclared tensor dtype at " + name);
            std::size_t elements = 1;
            if (value.shape.empty()) throw std::runtime_error("empty tensor shape for " + name);
            for (auto dimension : value.shape) {
                if (!dimension) throw std::runtime_error("zero tensor dimension");
                if (elements > std::numeric_limits<std::size_t>::max() / dimension) throw std::runtime_error("tensor shape product overflow for " + name);
                elements *= dimension;
            }
            const std::size_t element_size = f32 ? 4 : 2;
            if (elements > std::numeric_limits<std::size_t>::max() / element_size) throw std::runtime_error("tensor byte length overflow for " + name);
            const std::size_t expected_length = elements * element_size;
            if (value.length != expected_length) throw std::runtime_error("tensor byte length mismatch for " + name);
            value.data = weights_.at(value.offset, value.length);
        }
        for (auto& [name, value] : plan_.values) {
            if (value.lifetime != "activation" && value.lifetime != "logits" && value.lifetime != "state") throw std::runtime_error("invalid graph value lifetime " + value.lifetime);
            if (value.symbolic_shape.empty()) throw std::runtime_error("graph value " + name + " has empty shape");
            for (const auto& dimension : value.symbolic_shape) {
                auto symbolic = plan_.shapes.find(dimension);
                if (symbolic != plan_.shapes.end()) value.shape.push_back(symbolic->second);
                else if (!dimension.empty() && (std::isdigit(static_cast<unsigned char>(dimension[0])) || dimension[0] == '-')) {
                    const auto resolved = parse_size(dimension);
                    if (!resolved) throw std::runtime_error("invalid graph value dimension " + dimension);
                    value.shape.push_back(resolved);
                } else throw std::runtime_error("unknown symbolic dimension " + dimension);
            }
        }
        auto graph_value = [&](const std::string& name) -> const GraphValue& {
            auto found = plan_.values.find(name);
            if (found == plan_.values.end()) throw std::runtime_error("graph value is not declared: " + name);
            return found->second;
        };
        const auto& declared_logits = graph_value("logits");
        if (declared_logits.shape != std::vector<std::size_t>{vocab} || declared_logits.lifetime != "logits" || declared_logits.dtype != parse_dtype(plan_.output_dtype)) throw std::runtime_error("logits must be a vocab_size logits value");
        auto require_value = [&](const std::string& name, const std::vector<std::size_t>& expected, const std::string& kind) {
            const auto& value = graph_value(name);
            if (value.shape != expected) throw std::runtime_error(kind + " value shape mismatch");
            const bool logits = name == "logits";
            const DType expected_dtype = parse_dtype(logits ? plan_.output_dtype : plan_.activation_dtype);
            if (value.dtype != expected_dtype) throw std::runtime_error("operator " + kind + " value " + name + " violates declared dtype");
            const std::string expected_lifetime = logits ? "logits" : "activation";
            if (value.lifetime != expected_lifetime) throw std::runtime_error("operator " + kind + " value " + name + " violates declared lifetime");
        };
        std::unordered_set<std::string> defined, used, used_tensors;
        for (const auto& op : plan_.ops) {
            static const std::unordered_set<std::string> registry{"embedding", "position_embedding", "save", "rms_norm", "layer_norm", "attention_rope", "attention_rope_qkv_bias", "attention_causal", "add", "swiglu", "gelu_exact", "linear"};
            if (!registry.contains(op.kind)) throw std::runtime_error("unsupported operator " + op.kind);
            for (const auto& input : op.inputs) { if (!defined.contains(input)) throw std::runtime_error("operator dependency is unavailable: " + input); graph_value(input); used.insert(input); }
            for (const auto& output : op.outputs) { graph_value(output); if (!defined.insert(output).second) throw std::runtime_error("duplicate output definition " + output); }
            for (const auto& alias : op.tensors) used_tensors.insert(alias);
            if (op.kind == "embedding") { arity(op, 0, 1, 1); require_attributes(op, {}); require_shape(tensor(op.tensors[0]), {vocab, hidden}, op.tensors[0]); require_value(op.outputs[0], {hidden}, op.kind); }
            else if (op.kind == "position_embedding") { arity(op, 1, 1, 1); require_attributes(op, {}); require_shape(tensor(op.tensors[0]), {shape("context_size"), hidden}, op.tensors[0]); require_value(op.inputs[0], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind); }
            else if (op.kind == "save") { arity(op, 1, 1, 0); require_attributes(op, {}); require_value(op.inputs[0], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind); }
            else if (op.kind == "rms_norm") { arity(op, 1, 1, 1); require_attributes(op, {"epsilon"}); require_shape(tensor(op.tensors[0]), {hidden}, op.tensors[0]); attr_float(op, "epsilon"); require_value(op.inputs[0], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind); }
            else if (op.kind == "layer_norm") { arity(op, 1, 1, 2); require_attributes(op, {"epsilon"}); require_shape(tensor(op.tensors[0]), {hidden}, op.tensors[0]); require_shape(tensor(op.tensors[1]), {hidden}, op.tensors[1]); attr_float(op, "epsilon"); require_value(op.inputs[0], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind); }
            else if (op.kind == "attention_rope" || op.kind == "attention_causal") {
                arity(op, 1, 1, 4);
                if (op.kind == "attention_rope") require_attributes(op, {"cache", "heads", "kv_heads", "theta"});
                else require_attributes(op, {"cache", "heads", "kv_heads"});
                const auto heads = attr_size(op, "heads");
                const auto kv_heads = attr_size(op, "kv_heads");
                if (!heads || !kv_heads || hidden % heads || heads % kv_heads) throw std::runtime_error("invalid attention head shape");
                const auto head_dim = hidden / heads;
                require_shape(tensor(op.tensors[0]), {hidden, hidden}, op.tensors[0]);
                require_shape(tensor(op.tensors[1]), {kv_heads * head_dim, hidden}, op.tensors[1]);
                require_shape(tensor(op.tensors[2]), {kv_heads * head_dim, hidden}, op.tensors[2]);
                require_shape(tensor(op.tensors[3]), {hidden, hidden}, op.tensors[3]);
                const auto cache = attr(op, "cache");
                if (op.kind == "attention_rope") attr_float(op, "theta");
                if (!caches_.contains(cache)) { caches_.emplace(cache, Cache{{}, {}, 0, kv_heads, head_dim}); cache_order_.push_back(cache); }
                else throw std::runtime_error("KV cache identifier reused by multiple attention operators");
                require_value(op.inputs[0], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind);
            } else if (op.kind == "attention_rope_qkv_bias") {
                arity(op, 1, 1, 7);
                require_attributes(op, {"cache", "heads", "kv_heads", "rope_layout", "scale", "theta"});
                const auto heads = attr_size(op, "heads"), kv_heads = attr_size(op, "kv_heads");
                if (hidden % heads || heads % kv_heads) throw std::runtime_error("invalid attention head shape");
                const auto head_dim = hidden / heads, kv_size = kv_heads * head_dim;
                require_shape(tensor(op.tensors[0]), {hidden, hidden}, op.tensors[0]);
                require_shape(tensor(op.tensors[1]), {hidden}, op.tensors[1]);
                require_shape(tensor(op.tensors[2]), {kv_size, hidden}, op.tensors[2]);
                require_shape(tensor(op.tensors[3]), {kv_size}, op.tensors[3]);
                require_shape(tensor(op.tensors[4]), {kv_size, hidden}, op.tensors[4]);
                require_shape(tensor(op.tensors[5]), {kv_size}, op.tensors[5]);
                require_shape(tensor(op.tensors[6]), {hidden, hidden}, op.tensors[6]);
                if (attr(op, "rope_layout") != "half_split") throw std::runtime_error("unsupported RoPE layout");
                attr_float(op, "scale"); attr_float(op, "theta");
                const auto cache = attr(op, "cache");
                if (!caches_.contains(cache)) { caches_.emplace(cache, Cache{{}, {}, 0, kv_heads, head_dim}); cache_order_.push_back(cache); }
                else throw std::runtime_error("KV cache identifier reused by multiple attention operators");
                require_value(op.inputs[0], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind);
            } else if (op.kind == "add") { arity(op, 2, 1, 0); require_attributes(op, {}); require_value(op.inputs[0], {hidden}, op.kind); require_value(op.inputs[1], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind); }
            else if (op.kind == "swiglu") {
                arity(op, 1, 1, 3); require_attributes(op, {}); const auto intermediate = tensor(op.tensors[0]).shape.at(0);
                require_shape(tensor(op.tensors[0]), {intermediate, hidden}, op.tensors[0]); require_shape(tensor(op.tensors[1]), {intermediate, hidden}, op.tensors[1]); require_shape(tensor(op.tensors[2]), {hidden, intermediate}, op.tensors[2]);
                require_value(op.inputs[0], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind);
            } else if (op.kind == "gelu_exact") {
                arity(op, 1, 1, 4); require_attributes(op, {"formula"}); const auto intermediate = tensor(op.tensors[0]).shape.at(0);
                require_shape(tensor(op.tensors[0]), {intermediate, hidden}, op.tensors[0]); require_shape(tensor(op.tensors[1]), {intermediate}, op.tensors[1]); require_shape(tensor(op.tensors[2]), {hidden, intermediate}, op.tensors[2]); require_shape(tensor(op.tensors[3]), {hidden}, op.tensors[3]);
                if (attr(op, "formula") != "erf") throw std::runtime_error("unsupported GELU formula");
                require_value(op.inputs[0], {hidden}, op.kind); require_value(op.outputs[0], {hidden}, op.kind);
            } else if (op.kind == "linear") {
                if (op.tensors.size() != 1 && op.tensors.size() != 2) throw std::runtime_error("invalid linear operator arity");
                if (op.inputs.size() != 1 || op.outputs.size() != 1) throw std::runtime_error("invalid linear operator arity");
                if (op.attributes.empty()) require_attributes(op, {});
                else {
                    require_attributes(op, {"result_rounding"});
                    if (attr(op, "result_rounding") != "BF16" || op.outputs[0] != "logits" || plan_.output_dtype != "F32") throw std::runtime_error("unsupported linear result rounding");
                }
                const auto& weight = tensor(op.tensors[0]);
                if (weight.shape.size() != 2 || weight.shape[1] != hidden) throw std::runtime_error("tensor shape/orientation mismatch for " + op.tensors[0]);
                if (op.tensors.size() == 2) require_shape(tensor(op.tensors[1]), {weight.shape[0]}, op.tensors[1]);
                require_value(op.inputs[0], {weight.shape[1]}, op.kind); require_value(op.outputs[0], {weight.shape[0]}, op.kind);
                if (op.outputs[0] == "logits" && weight.shape[0] != vocab) throw std::runtime_error("LM head must produce exactly vocab_size logits");
            }
        }
        for (const auto& [alias, binding] : plan_.tensors) {
            (void)binding;
            if (!used_tensors.contains(alias)) throw std::runtime_error("unused plan tensor binding: " + alias);
        }
        if (defined.size() != plan_.values.size()) throw std::runtime_error("graph values are not defined");
        for (const auto& name : defined) if (name != "logits" && !used.contains(name)) throw std::runtime_error("graph values are unused: " + name);
        const auto& logits = graph_value("logits");
        if (!defined.contains("logits") || logits.shape != std::vector<std::size_t>{vocab} || logits.lifetime != "logits") throw std::runtime_error("logits must be a vocab_size logits value");
        validate_checkpoint_bindings(plan_, plan_.checkpoint_path);
        if (!plan_.canonical_serialization) throw std::runtime_error("noncanonical execution plan serialization");
        if (!plan_.canonical_digest) throw std::runtime_error("canonical plan identity mismatch");
        compiled_ops_.reserve(plan_.ops.size());
        for (const auto& op : plan_.ops) {
            CompiledOp compiled{&op, {}};
            compiled.tensors.reserve(op.tensors.size());
            for (const auto& alias : op.tensors) compiled.tensors.push_back(&tensor(alias));
            compiled_ops_.push_back(std::move(compiled));
        }
    }

    float value(const Tensor& tensor, std::size_t index) const {
        const std::size_t bytes_per_value = tensor.dtype == DType::F32 ? 4 : 2;
        if (index >= tensor.length / bytes_per_value) throw std::runtime_error("tensor element outside checkpoint");
        const auto* bytes = tensor.data + index * bytes_per_value;
        const auto byte = [&](std::size_t offset) { return std::to_integer<std::uint32_t>(bytes[offset]); };
        if (tensor.dtype == DType::F32) {
            const std::uint32_t bits = byte(0) | (byte(1) << 8U) | (byte(2) << 16U) | (byte(3) << 24U);
            return std::bit_cast<float>(bits);
        }
        return bf16_to_float(static_cast<std::uint16_t>(byte(0) | (byte(1) << 8U)));
    }
    float quantize(float input) const { return plan_.activation_dtype == "BF16" ? bf16_to_float(float_to_bf16(input)) : input; }
    float quantize_output(float input) const { return plan_.output_dtype == "BF16" ? bf16_to_float(float_to_bf16(input)) : input; }
    std::vector<float> linear(const std::vector<float>& input, const Tensor& weight, const Tensor* bias = nullptr, bool logits = false) {
        if (weight.shape.size() != 2 || weight.shape[1] != input.size()) throw std::runtime_error("native plan linear shape mismatch");
        ++matrix_passes_;
        std::vector<float> output(weight.shape[0]);
        const std::size_t rows = output.size(), columns = input.size();
        const std::size_t grain = std::min<std::size_t>(16, std::max<std::size_t>(1, rows / (parallel_.threads() * 4)));
        parallel_.run(rows, grain, [&](std::size_t begin, std::size_t end) {
            if (weight.dtype == DType::BF16 && reinterpret_cast<std::uintptr_t>(weight.data) % alignof(std::uint16_t) == 0) {
                const auto* weights = reinterpret_cast<const std::uint16_t*>(weight.data);
                std::size_t row = begin;
                // Adapted directly from portable_bf16_rows in strpot_native.cpp.
                for (; row + 4 <= end; row += 4) {
                    const auto* row0 = weights + row * columns;
                    const auto* row1 = row0 + columns;
                    const auto* row2 = row1 + columns;
                    const auto* row3 = row2 + columns;
                    float sum0 = bias ? value(*bias, row) : 0.0F;
                    float sum1 = bias ? value(*bias, row + 1) : 0.0F;
                    float sum2 = bias ? value(*bias, row + 2) : 0.0F;
                    float sum3 = bias ? value(*bias, row + 3) : 0.0F;
                    for (std::size_t column = 0; column < columns; ++column) {
                        const float item = input[column];
                        sum0 += bf16_to_float(row0[column]) * item;
                        sum1 += bf16_to_float(row1[column]) * item;
                        sum2 += bf16_to_float(row2[column]) * item;
                        sum3 += bf16_to_float(row3[column]) * item;
                    }
                    output[row] = logits ? quantize_output(sum0) : quantize(sum0);
                    output[row + 1] = logits ? quantize_output(sum1) : quantize(sum1);
                    output[row + 2] = logits ? quantize_output(sum2) : quantize(sum2);
                    output[row + 3] = logits ? quantize_output(sum3) : quantize(sum3);
                }
                for (; row < end; ++row) {
                    const auto* row_data = weights + row * columns;
                    float sum = bias ? value(*bias, row) : 0.0F;
                    for (std::size_t column = 0; column < columns; ++column) sum += bf16_to_float(row_data[column]) * input[column];
                    output[row] = logits ? quantize_output(sum) : quantize(sum);
                }
            } else {
                // Alignment-safe portable fallback for arbitrary tensor offsets.
                for (std::size_t row = begin; row < end; ++row) {
                    float sum = bias ? value(*bias, row) : 0.0F;
                    for (std::size_t column = 0; column < columns; ++column) sum += value(weight, row * columns + column) * input[column];
                    output[row] = logits ? quantize_output(sum) : quantize(sum);
                }
            }
        });
        return output;
    }
    std::vector<float> linear_block(const std::vector<float>& input, std::size_t positions, const Tensor& weight, const Tensor* bias = nullptr, bool logits = false) {
        if (weight.shape.size() != 2 || !positions || input.size() != positions * weight.shape[1]) throw std::runtime_error("native plan block linear shape mismatch");
        ++matrix_passes_;
        const std::size_t rows = weight.shape[0], columns = weight.shape[1];
        std::vector<float> output(positions * rows);
        const std::size_t grain = std::min<std::size_t>(16, std::max<std::size_t>(1, rows / (parallel_.threads() * 4)));
        parallel_.run(rows, grain, [&](std::size_t begin, std::size_t end) {
            if (weight.dtype == DType::BF16 && reinterpret_cast<std::uintptr_t>(weight.data) % alignof(std::uint16_t) == 0) {
                const auto* weights = reinterpret_cast<const std::uint16_t*>(weight.data);
                std::size_t row = begin;
                for (; row + 4 <= end; row += 4) {
                    const auto* row0 = weights + row * columns; const auto* row1 = row0 + columns; const auto* row2 = row1 + columns; const auto* row3 = row2 + columns;
                    std::size_t position = 0;
                    for (; position + 4 <= positions; position += 4) {
                        float sums[4][4]{};
                        for (std::size_t r = 0; r < 4; ++r) for (std::size_t p = 0; p < 4; ++p) sums[r][p] = bias ? value(*bias, row + r) : 0.0F;
                        for (std::size_t column = 0; column < columns; ++column) {
                            const float w0 = bf16_to_float(row0[column]), w1 = bf16_to_float(row1[column]), w2 = bf16_to_float(row2[column]), w3 = bf16_to_float(row3[column]);
                            for (std::size_t p = 0; p < 4; ++p) { const float item = input[(position + p) * columns + column]; sums[0][p] += w0 * item; sums[1][p] += w1 * item; sums[2][p] += w2 * item; sums[3][p] += w3 * item; }
                        }
                        for (std::size_t r = 0; r < 4; ++r) for (std::size_t p = 0; p < 4; ++p) output[(position + p) * rows + row + r] = logits ? quantize_output(sums[r][p]) : quantize(sums[r][p]);
                    }
                    for (; position < positions; ++position) {
                        float sums[4]{bias ? value(*bias, row) : 0.0F, bias ? value(*bias, row + 1) : 0.0F, bias ? value(*bias, row + 2) : 0.0F, bias ? value(*bias, row + 3) : 0.0F};
                        for (std::size_t column = 0; column < columns; ++column) { const float item = input[position * columns + column]; sums[0] += bf16_to_float(row0[column]) * item; sums[1] += bf16_to_float(row1[column]) * item; sums[2] += bf16_to_float(row2[column]) * item; sums[3] += bf16_to_float(row3[column]) * item; }
                        for (std::size_t r = 0; r < 4; ++r) output[position * rows + row + r] = logits ? quantize_output(sums[r]) : quantize(sums[r]);
                    }
                }
                for (; row < end; ++row) for (std::size_t position = 0; position < positions; ++position) {
                    float sum = bias ? value(*bias, row) : 0.0F; const auto* row_data = weights + row * columns;
                    for (std::size_t column = 0; column < columns; ++column) sum += bf16_to_float(row_data[column]) * input[position * columns + column];
                    output[position * rows + row] = logits ? quantize_output(sum) : quantize(sum);
                }
            } else {
                for (std::size_t row = begin; row < end; ++row) for (std::size_t position = 0; position < positions; ++position) {
                    float sum = bias ? value(*bias, row) : 0.0F;
                    for (std::size_t column = 0; column < columns; ++column) sum += value(weight, row * columns + column) * input[position * columns + column];
                    output[position * rows + row] = logits ? quantize_output(sum) : quantize(sum);
                }
            }
        });
        return output;
    }
    std::vector<float> normalize(const std::vector<float>& input, const Tensor& scale, const Tensor* bias, float epsilon) const {
        float mean = 0.0F;
        if (bias) { for (float item : input) mean += item; mean /= static_cast<float>(input.size()); }
        float variance = 0.0F;
        for (float item : input) { const float centered = item - mean; variance += centered * centered; }
        const float inverse = 1.0F / std::sqrt(variance / static_cast<float>(input.size()) + epsilon);
        std::vector<float> output(input.size());
        for (std::size_t index = 0; index < input.size(); ++index) {
            const float normalized = quantize((input[index] - mean) * inverse);
            output[index] = quantize(normalized * value(scale, index) + (bias ? value(*bias, index) : 0.0F));
        }
        return output;
    }
    std::vector<float> normalize_block(const std::vector<float>& input, std::size_t positions, const Tensor& scale, const Tensor* bias, float epsilon) const {
        if (!positions || scale.shape.size() != 1 || input.size() != positions * scale.shape[0]) throw std::runtime_error("native plan block normalization shape mismatch");
        const std::size_t width = scale.shape[0]; std::vector<float> output(input.size());
        for (std::size_t position = 0; position < positions; ++position) {
            const std::size_t base = position * width; float mean = 0.0F;
            if (bias) { for (std::size_t i = 0; i < width; ++i) mean += input[base + i]; mean /= static_cast<float>(width); }
            float variance = 0.0F; for (std::size_t i = 0; i < width; ++i) { const float centered = input[base + i] - mean; variance += centered * centered; }
            const float inverse = 1.0F / std::sqrt(variance / static_cast<float>(width) + epsilon);
            for (std::size_t i = 0; i < width; ++i) { const float normalized = quantize((input[base + i] - mean) * inverse); output[base + i] = quantize(normalized * value(scale, i) + (bias ? value(*bias, i) : 0.0F)); }
        }
        return output;
    }
    void rope(std::vector<float>& values, std::size_t heads, std::size_t head_dim, std::size_t position, float theta) const {
        const std::size_t half = head_dim / 2;
        if (!half || head_dim % 2) throw std::runtime_error("RoPE requires an even head dimension");
        for (std::size_t head = 0; head < heads; ++head) for (std::size_t index = 0; index < half; ++index) {
            const float angle = static_cast<float>(position) / std::pow(theta, static_cast<float>(2 * index) / static_cast<float>(head_dim));
            const std::size_t first_index = head * head_dim + index, second_index = first_index + half;
            const float first = values[first_index], second = values[second_index];
            const float cosine = quantize(std::cos(angle)), sine = quantize(std::sin(angle));
            const float first_cosine = quantize(first * cosine), second_sine = quantize(second * sine);
            const float second_cosine = quantize(second * cosine), first_sine = quantize(first * sine);
            values[first_index] = quantize(first_cosine - second_sine);
            values[second_index] = quantize(second_cosine + first_sine);
        }
    }
    std::vector<float> attention(const Op& op, const std::vector<const Tensor*>& bindings, const std::vector<float>& input, std::size_t position) {
        const auto heads = attr_size(op, "heads"), kv_heads = attr_size(op, "kv_heads"), head_dim = shape("hidden_size") / heads;
        const bool biased = op.kind == "attention_rope_qkv_bias";
        auto query = linear(input, *bindings[0], biased ? bindings[1] : nullptr);
        auto key = linear(input, *bindings[biased ? 2 : 1], biased ? bindings[3] : nullptr);
        auto val = linear(input, *bindings[biased ? 4 : 2], biased ? bindings[5] : nullptr);
        if (op.kind == "attention_rope" || biased) { const float theta = attr_float(op, "theta"); rope(query, heads, head_dim, position, theta); rope(key, kv_heads, head_dim, position, theta); }
        auto& cache = caches_.at(attr(op, "cache")); cache.key.insert(cache.key.end(), key.begin(), key.end()); cache.value.insert(cache.value.end(), val.begin(), val.end()); ++cache.length;
        std::vector<float> attended(shape("hidden_size")); const auto repeats = heads / kv_heads;
        for (std::size_t head = 0; head < heads; ++head) {
            const auto kv_head = head / repeats; std::vector<float> scores(cache.length); float maximum = -INFINITY;
            const float scale = biased ? attr_float(op, "scale") : 1.0F / std::sqrt(static_cast<float>(head_dim));
            for (std::size_t token = 0; token < cache.length; ++token) { float score = 0.0F; for (std::size_t dim = 0; dim < head_dim; ++dim) score += query[head * head_dim + dim] * cache.key[(token * kv_heads + kv_head) * head_dim + dim]; scores[token] = quantize(quantize(score) * scale); maximum = std::max(maximum, scores[token]); }
            float denominator = 0.0F; for (float& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (std::size_t dim = 0; dim < head_dim; ++dim) { float sum = 0.0F; for (std::size_t token = 0; token < cache.length; ++token) { const float probability = quantize(scores[token] / denominator); sum += probability * cache.value[(token * kv_heads + kv_head) * head_dim + dim]; } attended[head * head_dim + dim] = quantize(sum); }
        }
        return linear(attended, *bindings[biased ? 6 : 3]);
    }

    std::vector<float> attention_block(const Op& op, const std::vector<const Tensor*>& bindings, const std::vector<float>& input, std::size_t positions, std::size_t first_position) {
        const auto heads = attr_size(op, "heads"), kv_heads = attr_size(op, "kv_heads"), hidden = shape("hidden_size"), head_dim = hidden / heads;
        const bool biased = op.kind == "attention_rope_qkv_bias"; const std::size_t kv_width = kv_heads * head_dim;
        auto query = linear_block(input, positions, *bindings[0], biased ? bindings[1] : nullptr);
        auto key = linear_block(input, positions, *bindings[biased ? 2 : 1], biased ? bindings[3] : nullptr);
        auto val = linear_block(input, positions, *bindings[biased ? 4 : 2], biased ? bindings[5] : nullptr);
        if (op.kind == "attention_rope" || biased) {
            const float theta = attr_float(op, "theta");
            for (std::size_t p = 0; p < positions; ++p) {
                std::vector<float> q(query.begin() + static_cast<std::ptrdiff_t>(p * hidden), query.begin() + static_cast<std::ptrdiff_t>((p + 1) * hidden));
                std::vector<float> k(key.begin() + static_cast<std::ptrdiff_t>(p * kv_width), key.begin() + static_cast<std::ptrdiff_t>((p + 1) * kv_width));
                rope(q, heads, head_dim, first_position + p, theta); rope(k, kv_heads, head_dim, first_position + p, theta);
                std::copy(q.begin(), q.end(), query.begin() + static_cast<std::ptrdiff_t>(p * hidden)); std::copy(k.begin(), k.end(), key.begin() + static_cast<std::ptrdiff_t>(p * kv_width));
            }
        }
        auto& cache = caches_.at(attr(op, "cache")); const std::size_t old_length = cache.length;
        cache.key.insert(cache.key.end(), key.begin(), key.end()); cache.value.insert(cache.value.end(), val.begin(), val.end()); cache.length += positions;
        std::vector<float> attended(positions * hidden); const auto repeats = heads / kv_heads;
        const float scale = biased ? attr_float(op, "scale") : 1.0F / std::sqrt(static_cast<float>(head_dim));
        for (std::size_t p = 0; p < positions; ++p) for (std::size_t head = 0; head < heads; ++head) {
            const auto kv_head = head / repeats, visible = old_length + p + 1; std::vector<float> scores(visible); float maximum = -INFINITY;
            for (std::size_t token = 0; token < visible; ++token) { float score = 0.0F; for (std::size_t dim = 0; dim < head_dim; ++dim) score += query[p * hidden + head * head_dim + dim] * cache.key[(token * kv_heads + kv_head) * head_dim + dim]; scores[token] = quantize(quantize(score) * scale); maximum = std::max(maximum, scores[token]); }
            float denominator = 0.0F; for (float& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (std::size_t dim = 0; dim < head_dim; ++dim) { float sum = 0.0F; for (std::size_t token = 0; token < visible; ++token) { const float probability = quantize(scores[token] / denominator); sum += probability * cache.value[(token * kv_heads + kv_head) * head_dim + dim]; } attended[p * hidden + head * head_dim + dim] = quantize(sum); }
        }
        return linear_block(attended, positions, *bindings[biased ? 6 : 3]);
    }

    std::vector<float> forward_block_all(const std::vector<int>& tokens, std::size_t first_position) {
        const std::size_t positions = tokens.size(), hidden = shape("hidden_size");
        std::unordered_map<std::string, std::vector<float>> values;
        for (const auto& compiled : compiled_ops_) {
            const auto& op = *compiled.op; const auto& bindings = compiled.tensors;
            if (op.kind == "embedding") { const auto& weight = *bindings[0]; std::vector<float> output(positions * hidden); for (std::size_t p = 0; p < positions; ++p) for (std::size_t i = 0; i < hidden; ++i) output[p * hidden + i] = quantize(value(weight, static_cast<std::size_t>(tokens[p]) * hidden + i)); values[op.outputs[0]] = std::move(output); }
            else if (op.kind == "position_embedding") { auto output = values.at(op.inputs[0]); const auto& weight = *bindings[0]; for (std::size_t p = 0; p < positions; ++p) for (std::size_t i = 0; i < hidden; ++i) output[p * hidden + i] = quantize(output[p * hidden + i] + value(weight, (first_position + p) * hidden + i)); values[op.outputs[0]] = std::move(output); }
            else if (op.kind == "save") values[op.outputs[0]] = values.at(op.inputs[0]);
            else if (op.kind == "rms_norm") values[op.outputs[0]] = normalize_block(values.at(op.inputs[0]), positions, *bindings[0], nullptr, attr_float(op, "epsilon"));
            else if (op.kind == "layer_norm") values[op.outputs[0]] = normalize_block(values.at(op.inputs[0]), positions, *bindings[0], bindings[1], attr_float(op, "epsilon"));
            else if (op.kind == "attention_rope" || op.kind == "attention_rope_qkv_bias" || op.kind == "attention_causal") values[op.outputs[0]] = attention_block(op, bindings, values.at(op.inputs[0]), positions, first_position);
            else if (op.kind == "add") { auto output = values.at(op.inputs[0]); const auto& other = values.at(op.inputs[1]); if (output.size() != other.size()) throw std::runtime_error("add shape mismatch"); for (std::size_t i = 0; i < output.size(); ++i) output[i] = quantize(output[i] + other[i]); values[op.outputs[0]] = std::move(output); }
            else if (op.kind == "swiglu") { auto gate = linear_block(values.at(op.inputs[0]), positions, *bindings[0]); auto up = linear_block(values.at(op.inputs[0]), positions, *bindings[1]); for (std::size_t i = 0; i < gate.size(); ++i) { const float activated = quantize(gate[i] / (1.0F + std::exp(-gate[i]))); gate[i] = quantize(activated * up[i]); } values[op.outputs[0]] = linear_block(gate, positions, *bindings[2]); }
            else if (op.kind == "gelu_exact") { auto intermediate = linear_block(values.at(op.inputs[0]), positions, *bindings[0], bindings[1]); for (float& item : intermediate) item = quantize(0.5F * item * (1.0F + std::erf(item / std::sqrt(2.0F)))); values[op.outputs[0]] = linear_block(intermediate, positions, *bindings[2], bindings[3]); }
            else if (op.kind == "linear") { const bool logits = op.outputs[0] == "logits"; auto output = linear_block(values.at(op.inputs[0]), positions, *bindings[0], bindings.size() == 2 ? bindings[1] : nullptr, logits); if (op.attributes.contains("result_rounding")) for (float& item : output) item = bf16_to_float(float_to_bf16(item)); values[op.outputs[0]] = std::move(output); }
        }
        executed_ += positions * compiled_ops_.size();
        for (std::size_t p = 0; p < positions; ++p) for (const auto& compiled : compiled_ops_) trace_.push_back(compiled.op->kind);
        const auto& all_logits = values.at("logits");
        return all_logits;
    }

    std::vector<float> forward_block(const std::vector<int>& tokens, std::size_t first_position) {
        auto all_logits = forward_block_all(tokens, first_position); const std::size_t vocab = shape("vocab_size");
        return std::vector<float>(all_logits.end() - static_cast<std::ptrdiff_t>(vocab), all_logits.end());
    }

    std::vector<float> forward(int token, std::size_t position) {
        std::unordered_map<std::string, std::vector<float>> values;
        for (const auto& compiled : compiled_ops_) {
            const auto& op = *compiled.op;
            const auto& bindings = compiled.tensors;
            ++executed_;
            trace_.push_back(op.kind);
            if (op.kind == "embedding") { const auto& weight = *bindings[0]; if (token < 0 || static_cast<std::size_t>(token) >= weight.shape[0]) throw std::runtime_error("token outside embedding table"); std::vector<float> output(weight.shape[1]); for (std::size_t i = 0; i < output.size(); ++i) output[i] = quantize(value(weight, static_cast<std::size_t>(token) * output.size() + i)); values[op.outputs[0]] = std::move(output); }
            else if (op.kind == "position_embedding") { auto output = values.at(op.inputs[0]); const auto& weight = *bindings[0]; for (std::size_t i = 0; i < output.size(); ++i) output[i] = quantize(output[i] + value(weight, position * output.size() + i)); values[op.outputs[0]] = std::move(output); }
            else if (op.kind == "save") values[op.outputs[0]] = values.at(op.inputs[0]);
            else if (op.kind == "rms_norm") values[op.outputs[0]] = normalize(values.at(op.inputs[0]), *bindings[0], nullptr, attr_float(op, "epsilon"));
            else if (op.kind == "layer_norm") values[op.outputs[0]] = normalize(values.at(op.inputs[0]), *bindings[0], bindings[1], attr_float(op, "epsilon"));
            else if (op.kind == "attention_rope" || op.kind == "attention_rope_qkv_bias" || op.kind == "attention_causal") values[op.outputs[0]] = attention(op, bindings, values.at(op.inputs[0]), position);
            else if (op.kind == "add") { auto output = values.at(op.inputs[0]); const auto& other = values.at(op.inputs[1]); if (output.size() != other.size()) throw std::runtime_error("add shape mismatch"); for (std::size_t i = 0; i < output.size(); ++i) output[i] = quantize(output[i] + other[i]); values[op.outputs[0]] = std::move(output); }
            else if (op.kind == "swiglu") { auto gate = linear(values.at(op.inputs[0]), *bindings[0]); auto up = linear(values.at(op.inputs[0]), *bindings[1]); for (std::size_t i = 0; i < gate.size(); ++i) { const float activated = quantize(gate[i] / (1.0F + std::exp(-gate[i]))); gate[i] = quantize(activated * up[i]); } values[op.outputs[0]] = linear(gate, *bindings[2]); }
            else if (op.kind == "gelu_exact") { auto hidden = linear(values.at(op.inputs[0]), *bindings[0], bindings[1]); for (float& item : hidden) item = quantize(0.5F * item * (1.0F + std::erf(item / std::sqrt(2.0F)))); values[op.outputs[0]] = linear(hidden, *bindings[2], bindings[3]); }
            else if (op.kind == "linear") { const bool logits = op.outputs[0] == "logits"; auto output = linear(values.at(op.inputs[0]), *bindings[0], bindings.size() == 2 ? bindings[1] : nullptr, logits); if (op.attributes.contains("result_rounding")) for (float& item : output) item = bf16_to_float(float_to_bf16(item)); values[op.outputs[0]] = std::move(output); }
        }
        return values.at("logits");
    }

    Plan plan_; MappedFile weights_; Parallel parallel_; std::unordered_map<std::string, Cache> caches_; std::vector<std::string> cache_order_; std::vector<CompiledOp> compiled_ops_; std::size_t position_{}; std::size_t executed_{}; std::size_t matrix_passes_{}; std::size_t runtime_tensor_lookups_{}; std::size_t prefill_matrix_passes_{}; std::size_t prefill_physical_width_{}; double prefill_seconds_{}; std::vector<std::string> trace_; std::vector<std::vector<float>> frontier_logits_; std::vector<double> decode_intervals_;
};

std::vector<int> parse_tokens(const std::string& value) {
    if (value.empty() || value.front() == ',' || value.back() == ',') throw std::runtime_error("malformed token list");
    std::vector<int> result;
    std::size_t start = 0;
    while (start < value.size()) {
        const auto end = value.find(',', start);
        const auto item = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        const auto parsed = parse_size(item);
        if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::runtime_error("token value out of range");
        result.push_back(static_cast<int>(parsed));
        if (end == std::string::npos) break;
        start = end + 1;
        if (start == value.size() || value[start] == ',') throw std::runtime_error("malformed token list");
    }
    return result;
}
std::string json_string(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result{"\""};
    for (const unsigned char item : value) {
        switch (item) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (item < 0x20U) { result += "\\u00"; result += hex[item >> 4U]; result += hex[item & 0x0fU]; }
                else result.push_back(static_cast<char>(item));
        }
    }
    result += '"';
    return result;
}
void print_strings(const std::vector<std::string>& values) { std::cout << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << json_string(values[i]); } std::cout << ']'; }
template <typename T> void print_values(const std::vector<T>& values) { std::cout << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; if constexpr (std::is_floating_point_v<T>) { if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite native result"); std::cout << std::setprecision(std::numeric_limits<T>::max_digits10); } std::cout << values[i]; } std::cout << ']'; }
template <typename T> void print_nested(const std::vector<std::vector<T>>& values) { std::cout << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; print_values(values[i]); } std::cout << ']'; }

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: strpot-native-plan PLAN CHECKPOINT TOKENS MAX_NEW THREADS");
        const auto threads = parse_size(argv[5]);
        if (!threads || threads > MAX_PARALLEL_THREADS) throw std::runtime_error("threads exceed safe bound");
        if (std::fesetround(FE_TONEAREST) != 0 || std::fegetround() != FE_TONEAREST) throw std::runtime_error("cannot establish round-to-nearest-even");
        Executor executor(read_plan(argv[1], argv[2]), argv[2], threads);
        const auto tokens = executor.generate(parse_tokens(argv[3]), parse_size(argv[4]));
        const auto cache_lengths = executor.cache_lengths(); const auto& plan = executor.plan();
        std::cout << "{\"engine\":" << json_string("strpot-native-plan") << ",\"architecture_id\":" << json_string(plan.architecture_id) << ",\"config_identity\":" << json_string(plan.config_identity) << ",\"weight_dtype\":" << json_string(plan.weight_dtype) << ",\"kernel_family\":" << json_string(executor.kernel_family()) << ",\"tokens\":"; print_values(tokens); std::cout << ",\"executed_operators\":" << executor.executed() << ",\"matrix_passes\":" << executor.matrix_passes() << ",\"parallel_dispatches\":" << executor.parallel_dispatches() << ",\"threaded_matrix_dispatches\":" << executor.threaded_matrix_dispatches() << ",\"runtime_tensor_lookups\":" << executor.runtime_tensor_lookups() << ",\"prefill_seconds\":" << std::setprecision(17) << executor.prefill_seconds() << ",\"prefill_matrix_passes\":" << executor.prefill_matrix_passes() << ",\"prefill_physical_width\":" << executor.prefill_physical_width() << ",\"kv_cache_lengths\":"; print_values(cache_lengths); std::cout << ",\"frontier_logits\":"; print_nested(executor.frontier_logits()); std::cout << ",\"inter_token_seconds\":"; print_values(executor.decode_intervals()); std::cout << ",\"kv_cache_keys\":"; print_nested(executor.cache_keys()); std::cout << ",\"kv_cache_values\":"; print_nested(executor.cache_values()); std::cout << ",\"operator_trace\":"; print_strings(executor.trace());
        std::cout << ",\"threads\":" << executor.threads() << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
