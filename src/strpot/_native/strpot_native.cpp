#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strpot {

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

float round_bf16(float value) { return bf16_to_float(float_to_bf16(value)); }

using Bf16RowsKernel = void (*)(
    const std::uint16_t*, const float*, float*, std::size_t, std::size_t, std::size_t
);
using F32RowsKernel = void (*)(
    const float*, const float*, float*, std::size_t, std::size_t, std::size_t
);
using Bf16BlockKernel = void (*)(
    const std::uint16_t*,
    const float*,
    float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);
using F32BlockKernel = void (*)(
    const float*,
    const float*,
    float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);
using Bf16PanelKernel = void (*)(
    const std::uint16_t*,
    const float*,
    float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);
using F32PanelKernel = void (*)(
    const float*,
    const float*,
    float*,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

struct KernelSet {
    const char* name;
    Bf16RowsKernel bf16_rows;
    F32RowsKernel f32_rows;
    Bf16BlockKernel bf16_block;
    F32BlockKernel f32_block;
    Bf16PanelKernel bf16_panel;
    F32PanelKernel f32_panel;
};

void portable_bf16_rows(
    const std::uint16_t* weights,
    const float* input,
    float* output,
    std::size_t columns,
    std::size_t begin,
    std::size_t end
) {
    std::size_t row = begin;
    for (; row + 4 <= end; row += 4) {
        const auto* row0 = weights + row * columns;
        const auto* row1 = row0 + columns;
        const auto* row2 = row1 + columns;
        const auto* row3 = row2 + columns;
        float sum0 = 0.0F;
        float sum1 = 0.0F;
        float sum2 = 0.0F;
        float sum3 = 0.0F;
        for (std::size_t column = 0; column < columns; ++column) {
            const float value = input[column];
            sum0 += bf16_to_float(row0[column]) * value;
            sum1 += bf16_to_float(row1[column]) * value;
            sum2 += bf16_to_float(row2[column]) * value;
            sum3 += bf16_to_float(row3[column]) * value;
        }
        output[row] = round_bf16(sum0);
        output[row + 1] = round_bf16(sum1);
        output[row + 2] = round_bf16(sum2);
        output[row + 3] = round_bf16(sum3);
    }
    for (; row < end; ++row) {
        const auto* row_data = weights + row * columns;
        float sum = 0.0F;
        for (std::size_t column = 0; column < columns; ++column) {
            sum += bf16_to_float(row_data[column]) * input[column];
        }
        output[row] = round_bf16(sum);
    }
}

void portable_f32_rows(
    const float* weights,
    const float* input,
    float* output,
    std::size_t columns,
    std::size_t begin,
    std::size_t end
) {
    std::size_t row = begin;
    for (; row + 4 <= end; row += 4) {
        const auto* row0 = weights + row * columns;
        const auto* row1 = row0 + columns;
        const auto* row2 = row1 + columns;
        const auto* row3 = row2 + columns;
        float sum0 = 0.0F;
        float sum1 = 0.0F;
        float sum2 = 0.0F;
        float sum3 = 0.0F;
        for (std::size_t column = 0; column < columns; ++column) {
            const float value = input[column];
            sum0 += row0[column] * value;
            sum1 += row1[column] * value;
            sum2 += row2[column] * value;
            sum3 += row3[column] * value;
        }
        output[row] = sum0;
        output[row + 1] = sum1;
        output[row + 2] = sum2;
        output[row + 3] = sum3;
    }
    for (; row < end; ++row) {
        const auto* row_data = weights + row * columns;
        float sum = 0.0F;
        for (std::size_t column = 0; column < columns; ++column) {
            sum += row_data[column] * input[column];
        }
        output[row] = sum;
    }
}

void portable_bf16_block(
    const std::uint16_t* weights,
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    std::size_t positions,
    std::size_t begin,
    std::size_t end
) {
    std::size_t row = begin;
    for (; row + 4 <= end; row += 4) {
        const auto* row0 = weights + row * columns;
        const auto* row1 = row0 + columns;
        const auto* row2 = row1 + columns;
        const auto* row3 = row2 + columns;
        std::size_t position = 0;
        for (; position + 4 <= positions; position += 4) {
            float sums[4][4]{};
            const auto* input0 = input + position * columns;
            const auto* input1 = input0 + columns;
            const auto* input2 = input1 + columns;
            const auto* input3 = input2 + columns;
            for (std::size_t column = 0; column < columns; ++column) {
                const float weight0 = bf16_to_float(row0[column]);
                const float weight1 = bf16_to_float(row1[column]);
                const float weight2 = bf16_to_float(row2[column]);
                const float weight3 = bf16_to_float(row3[column]);
                const float value0 = input0[column];
                const float value1 = input1[column];
                const float value2 = input2[column];
                const float value3 = input3[column];
                sums[0][0] += weight0 * value0;
                sums[0][1] += weight0 * value1;
                sums[0][2] += weight0 * value2;
                sums[0][3] += weight0 * value3;
                sums[1][0] += weight1 * value0;
                sums[1][1] += weight1 * value1;
                sums[1][2] += weight1 * value2;
                sums[1][3] += weight1 * value3;
                sums[2][0] += weight2 * value0;
                sums[2][1] += weight2 * value1;
                sums[2][2] += weight2 * value2;
                sums[2][3] += weight2 * value3;
                sums[3][0] += weight3 * value0;
                sums[3][1] += weight3 * value1;
                sums[3][2] += weight3 * value2;
                sums[3][3] += weight3 * value3;
            }
            for (std::size_t output_row = 0; output_row < 4; ++output_row) {
                for (std::size_t output_position = 0; output_position < 4; ++output_position) {
                    output[(position + output_position) * rows + row + output_row] =
                        round_bf16(sums[output_row][output_position]);
                }
            }
        }
        for (; position < positions; ++position) {
            const auto* values = input + position * columns;
            float sum0 = 0.0F;
            float sum1 = 0.0F;
            float sum2 = 0.0F;
            float sum3 = 0.0F;
            for (std::size_t column = 0; column < columns; ++column) {
                const float value = values[column];
                sum0 += bf16_to_float(row0[column]) * value;
                sum1 += bf16_to_float(row1[column]) * value;
                sum2 += bf16_to_float(row2[column]) * value;
                sum3 += bf16_to_float(row3[column]) * value;
            }
            const std::size_t target = position * rows + row;
            output[target] = round_bf16(sum0);
            output[target + 1] = round_bf16(sum1);
            output[target + 2] = round_bf16(sum2);
            output[target + 3] = round_bf16(sum3);
        }
    }
    for (; row < end; ++row) {
        const auto* row_data = weights + row * columns;
        std::size_t position = 0;
        for (; position + 4 <= positions; position += 4) {
            const auto* input0 = input + position * columns;
            const auto* input1 = input0 + columns;
            const auto* input2 = input1 + columns;
            const auto* input3 = input2 + columns;
            float sum0 = 0.0F;
            float sum1 = 0.0F;
            float sum2 = 0.0F;
            float sum3 = 0.0F;
            for (std::size_t column = 0; column < columns; ++column) {
                const float weight = bf16_to_float(row_data[column]);
                sum0 += weight * input0[column];
                sum1 += weight * input1[column];
                sum2 += weight * input2[column];
                sum3 += weight * input3[column];
            }
            output[position * rows + row] = round_bf16(sum0);
            output[(position + 1) * rows + row] = round_bf16(sum1);
            output[(position + 2) * rows + row] = round_bf16(sum2);
            output[(position + 3) * rows + row] = round_bf16(sum3);
        }
        for (; position < positions; ++position) {
            const auto* values = input + position * columns;
            float sum = 0.0F;
            for (std::size_t column = 0; column < columns; ++column) {
                sum += bf16_to_float(row_data[column]) * values[column];
            }
            output[position * rows + row] = round_bf16(sum);
        }
    }
}

void portable_f32_block(
    const float* weights,
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t columns,
    std::size_t positions,
    std::size_t begin,
    std::size_t end
) {
    for (std::size_t row = begin; row < end; ++row) {
        const auto* row_data = weights + row * columns;
        for (std::size_t position = 0; position < positions; ++position) {
            const auto* values = input + position * columns;
            float sum = 0.0F;
            for (std::size_t column = 0; column < columns; ++column) {
                sum += row_data[column] * values[column];
            }
            output[position * rows + row] = sum;
        }
    }
}

struct PanelSums16 {
    float s0{};
    float s1{};
    float s2{};
    float s3{};
    float s4{};
    float s5{};
    float s6{};
    float s7{};
    float s8{};
    float s9{};
    float s10{};
    float s11{};
    float s12{};
    float s13{};
    float s14{};
    float s15{};
};

inline void accumulate_panel_pair(
    PanelSums16& first,
    PanelSums16& second,
    float first_weight,
    float second_weight,
    const float* values
) {
    first.s0 += first_weight * values[0];
    second.s0 += second_weight * values[0];
    first.s1 += first_weight * values[1];
    second.s1 += second_weight * values[1];
    first.s2 += first_weight * values[2];
    second.s2 += second_weight * values[2];
    first.s3 += first_weight * values[3];
    second.s3 += second_weight * values[3];
    first.s4 += first_weight * values[4];
    second.s4 += second_weight * values[4];
    first.s5 += first_weight * values[5];
    second.s5 += second_weight * values[5];
    first.s6 += first_weight * values[6];
    second.s6 += second_weight * values[6];
    first.s7 += first_weight * values[7];
    second.s7 += second_weight * values[7];
    first.s8 += first_weight * values[8];
    second.s8 += second_weight * values[8];
    first.s9 += first_weight * values[9];
    second.s9 += second_weight * values[9];
    first.s10 += first_weight * values[10];
    second.s10 += second_weight * values[10];
    first.s11 += first_weight * values[11];
    second.s11 += second_weight * values[11];
    first.s12 += first_weight * values[12];
    second.s12 += second_weight * values[12];
    first.s13 += first_weight * values[13];
    second.s13 += second_weight * values[13];
    first.s14 += first_weight * values[14];
    second.s14 += second_weight * values[14];
    first.s15 += first_weight * values[15];
    second.s15 += second_weight * values[15];
}

inline void store_panel_sums(
    const PanelSums16& sums,
    float* output,
    std::size_t rows,
    std::size_t row,
    std::size_t position,
    std::size_t count
) {
    const float values[16] = {
        sums.s0, sums.s1, sums.s2, sums.s3, sums.s4, sums.s5, sums.s6, sums.s7,
        sums.s8, sums.s9, sums.s10, sums.s11, sums.s12, sums.s13, sums.s14, sums.s15,
    };
    for (std::size_t lane = 0; lane < count; ++lane) {
        output[(position + lane) * rows + row] = round_bf16(values[lane]);
    }
}

void portable_bf16_panel(
    const std::uint16_t* weights,
    const float* panel,
    float* output,
    std::size_t rows,
    std::size_t columns,
    std::size_t positions,
    std::size_t panel_stride,
    std::size_t begin,
    std::size_t end
) {
    constexpr std::size_t width = 16;
    for (std::size_t row = begin; row < end; row += 4) {
        const std::size_t row1 = std::min(row + 1, end - 1);
        const std::size_t row2 = std::min(row + 2, end - 1);
        const std::size_t row3 = std::min(row + 3, end - 1);
        const auto* weights0 = weights + row * columns;
        const auto* weights1 = weights + row1 * columns;
        const auto* weights2 = weights + row2 * columns;
        const auto* weights3 = weights + row3 * columns;
        for (std::size_t position = 0; position < positions; position += width) {
            const std::size_t count = std::min(width, positions - position);
            PanelSums16 sums0;
            PanelSums16 sums1;
            PanelSums16 sums2;
            PanelSums16 sums3;
            for (std::size_t column = 0; column < columns; ++column) {
                const float* values = panel + column * panel_stride + position;
                accumulate_panel_pair(
                    sums0,
                    sums1,
                    bf16_to_float(weights0[column]),
                    bf16_to_float(weights1[column]),
                    values
                );
                accumulate_panel_pair(
                    sums2,
                    sums3,
                    bf16_to_float(weights2[column]),
                    bf16_to_float(weights3[column]),
                    values
                );
            }
            store_panel_sums(sums0, output, rows, row, position, count);
            if (row1 != row) {
                store_panel_sums(sums1, output, rows, row1, position, count);
            }
            if (row2 != row1) {
                store_panel_sums(sums2, output, rows, row2, position, count);
            }
            if (row3 != row2) {
                store_panel_sums(sums3, output, rows, row3, position, count);
            }
        }
    }
}

void portable_f32_panel(
    const float* weights,
    const float* panel,
    float* output,
    std::size_t rows,
    std::size_t columns,
    std::size_t positions,
    std::size_t panel_stride,
    std::size_t begin,
    std::size_t end
) {
    constexpr std::size_t width = 16;
    for (std::size_t row = begin; row < end; ++row) {
        const auto* row_data = weights + row * columns;
        for (std::size_t position = 0; position < positions; position += width) {
            const std::size_t count = std::min(width, positions - position);
            float sum0 = 0.0F;
            float sum1 = 0.0F;
            float sum2 = 0.0F;
            float sum3 = 0.0F;
            float sum4 = 0.0F;
            float sum5 = 0.0F;
            float sum6 = 0.0F;
            float sum7 = 0.0F;
            float sum8 = 0.0F;
            float sum9 = 0.0F;
            float sum10 = 0.0F;
            float sum11 = 0.0F;
            float sum12 = 0.0F;
            float sum13 = 0.0F;
            float sum14 = 0.0F;
            float sum15 = 0.0F;
            for (std::size_t column = 0; column < columns; ++column) {
                const float weight = row_data[column];
                const float* values = panel + column * panel_stride + position;
                sum0 += weight * values[0];
                sum1 += weight * values[1];
                sum2 += weight * values[2];
                sum3 += weight * values[3];
                sum4 += weight * values[4];
                sum5 += weight * values[5];
                sum6 += weight * values[6];
                sum7 += weight * values[7];
                sum8 += weight * values[8];
                sum9 += weight * values[9];
                sum10 += weight * values[10];
                sum11 += weight * values[11];
                sum12 += weight * values[12];
                sum13 += weight * values[13];
                sum14 += weight * values[14];
                sum15 += weight * values[15];
            }
            const float sums[width] = {
                sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7,
                sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15,
            };
            for (std::size_t lane = 0; lane < count; ++lane) {
                output[(position + lane) * rows + row] = sums[lane];
            }
        }
    }
}

KernelSet portable_kernel_set() {
    return KernelSet{
        "portable-tiled",
        portable_bf16_rows,
        portable_f32_rows,
        portable_bf16_block,
        portable_f32_block,
        portable_bf16_panel,
        portable_f32_panel,
    };
}

enum class DType { BF16, F32 };

struct Tensor {
    DType dtype;
    std::size_t offset;
    std::size_t byte_size;
    std::vector<std::size_t> shape;
};

struct Config {
    std::size_t hidden_size{};
    std::size_t intermediate_size{};
    std::size_t num_attention_heads{};
    std::size_t num_key_value_heads{};
    std::size_t num_hidden_layers{};
    std::size_t vocab_size{};
    std::size_t max_position_embeddings{};
    float rms_norm_eps{};
    float rope_theta{};
    int bos_token_id{};
    int eos_token_id{};
};

struct Descriptor {
    Config config;
    std::unordered_map<std::string, Tensor> tensors;
};

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, separator)) {
        parts.push_back(part);
    }
    return parts;
}

Descriptor read_descriptor(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open native model descriptor");
    }
    std::string line;
    if (!std::getline(stream, line) || line != "STRPOT_NATIVE_V1") {
        throw std::runtime_error("unsupported native model descriptor");
    }
    Descriptor descriptor;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split(line, '\t');
        if (fields.size() == 3 && fields[0] == "config") {
            const auto& key = fields[1];
            const auto& value = fields[2];
            if (key == "hidden_size") descriptor.config.hidden_size = std::stoull(value);
            else if (key == "intermediate_size") descriptor.config.intermediate_size = std::stoull(value);
            else if (key == "num_attention_heads") descriptor.config.num_attention_heads = std::stoull(value);
            else if (key == "num_key_value_heads") descriptor.config.num_key_value_heads = std::stoull(value);
            else if (key == "num_hidden_layers") descriptor.config.num_hidden_layers = std::stoull(value);
            else if (key == "vocab_size") descriptor.config.vocab_size = std::stoull(value);
            else if (key == "max_position_embeddings") descriptor.config.max_position_embeddings = std::stoull(value);
            else if (key == "rms_norm_eps") descriptor.config.rms_norm_eps = std::stof(value);
            else if (key == "rope_theta") descriptor.config.rope_theta = std::stof(value);
            else if (key == "bos_token_id") descriptor.config.bos_token_id = std::stoi(value);
            else if (key == "eos_token_id") descriptor.config.eos_token_id = std::stoi(value);
            continue;
        }
        if (fields.size() != 5 || fields[0] != "tensor") {
            throw std::runtime_error("invalid native descriptor record");
        }
        DType dtype;
        if (fields[2] == "BF16") dtype = DType::BF16;
        else if (fields[2] == "F32") dtype = DType::F32;
        else throw std::runtime_error("native engine does not support tensor dtype " + fields[2]);
        std::vector<std::size_t> shape;
        for (const auto& dimension : split(fields[4], ',')) {
            const std::size_t parsed = std::stoull(dimension);
            if (parsed == 0 || parsed > 0x7fffffffULL || shape.size() >= 8) {
                throw std::runtime_error("invalid or unbounded tensor dimensions");
            }
            shape.push_back(parsed);
        }
        if (shape.empty()) throw std::runtime_error("invalid tensor shape");
        std::size_t byte_size = dtype == DType::BF16 ? 2 : 4;
        for (const std::size_t dimension : shape) {
            if (byte_size > std::numeric_limits<std::size_t>::max() / dimension) {
                throw std::runtime_error("tensor shape byte size overflow");
            }
            byte_size *= dimension;
        }
        const auto inserted = descriptor.tensors.emplace(
            fields[1], Tensor{dtype, std::stoull(fields[3]), byte_size, std::move(shape)}
        );
        if (!inserted.second) throw std::runtime_error("duplicate tensor descriptor");
    }
    const auto& config = descriptor.config;
    if (config.hidden_size == 0 || config.num_hidden_layers == 0 ||
        config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
        config.hidden_size % config.num_attention_heads != 0 ||
        config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("invalid native Llama configuration");
    }
    return descriptor;
}

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        descriptor_ = open(path.c_str(), O_RDONLY);
        if (descriptor_ < 0) throw std::runtime_error("cannot open checkpoint");
        struct stat status {};
        if (fstat(descriptor_, &status) != 0) {
            close(descriptor_);
            throw std::runtime_error("cannot stat checkpoint");
        }
        size_ = static_cast<std::size_t>(status.st_size);
        data_ = static_cast<const std::byte*>(
            mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0)
        );
        if (data_ == MAP_FAILED) {
            close(descriptor_);
            data_ = nullptr;
            throw std::runtime_error("cannot map checkpoint");
        }
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    ~MappedFile() {
        if (data_ != nullptr) munmap(const_cast<std::byte*>(data_), size_);
        if (descriptor_ >= 0) close(descriptor_);
    }

    const std::byte* at(std::size_t offset) const {
        if (offset >= size_) throw std::runtime_error("tensor offset outside checkpoint");
        return data_ + offset;
    }

    std::size_t size() const { return size_; }

private:
    int descriptor_{-1};
    std::size_t size_{};
    const std::byte* data_{};
};

class Parallel {
public:
    explicit Parallel(std::size_t threads) : completion_(threads) {
        const std::size_t worker_count = threads > 1 ? threads - 1 : 0;
        workers_.reserve(worker_count);
        for (std::size_t index = 0; index < worker_count; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
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

    void run(
        std::size_t total,
        std::size_t grain,
        std::function<void(std::size_t, std::size_t)> function
    ) {
        ++dispatches_;
        if (workers_.empty() || total <= grain) {
            function(0, total);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            function_ = std::move(function);
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
    void reset_dispatches() { dispatches_ = 0; }

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
                start_.wait(lock, [this, observed] {
                    return stopping_ || generation_ != observed;
                });
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
};

class LlamaModel {
    struct LayerPlan {
        const Tensor* input_norm;
        const Tensor* query;
        const Tensor* key;
        const Tensor* value;
        const Tensor* attention_output;
        const Tensor* post_attention_norm;
        const Tensor* gate;
        const Tensor* up;
        const Tensor* down;
    };

public:
    LlamaModel(Descriptor descriptor, const std::string& checkpoint, std::size_t threads)
        : descriptor_(std::move(descriptor)),
          weights_(checkpoint),
          parallel_(threads),
          threads_(threads) {
        validate_tensor_ranges();
        const auto& config = descriptor_.config;
        head_dim_ = config.hidden_size / config.num_attention_heads;
        initialize_caches(caches_);
        embedding_weight_ = &tensor("model.embed_tokens.weight");
        final_norm_ = &tensor("model.norm.weight");
        lm_head_ = &tensor("lm_head.weight");
        layers_.reserve(config.num_hidden_layers);
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const std::string prefix = "model.layers." + std::to_string(layer);
            layers_.push_back(LayerPlan{
                &tensor(prefix + ".input_layernorm.weight"),
                &tensor(prefix + ".self_attn.q_proj.weight"),
                &tensor(prefix + ".self_attn.k_proj.weight"),
                &tensor(prefix + ".self_attn.v_proj.weight"),
                &tensor(prefix + ".self_attn.o_proj.weight"),
                &tensor(prefix + ".post_attention_layernorm.weight"),
                &tensor(prefix + ".mlp.gate_proj.weight"),
                &tensor(prefix + ".mlp.up_proj.weight"),
                &tensor(prefix + ".mlp.down_proj.weight"),
            });
        }
        const std::size_t half = head_dim_ / 2;
        rope_cos_.resize(config.max_position_embeddings * half);
        rope_sin_.resize(config.max_position_embeddings * half);
        attention_scores_.resize(config.max_position_embeddings);
        for (std::size_t position = 0; position < config.max_position_embeddings; ++position) {
            for (std::size_t index = 0; index < half; ++index) {
                const float frequency = 1.0F / std::pow(
                    config.rope_theta,
                    static_cast<float>(2 * index) / static_cast<float>(head_dim_)
                );
                const float angle = static_cast<float>(position) * frequency;
                const std::size_t table_index = position * half + index;
                rope_cos_[table_index] = quantize(
                    std::cos(angle), embedding_weight_->dtype
                );
                rope_sin_[table_index] = quantize(
                    std::sin(angle), embedding_weight_->dtype
                );
            }
        }
        tensor_lookups_ = 0;
    }

    struct Generation {
        std::vector<int> tokens;
        std::vector<double> inter_token_seconds;
        double prefill_seconds{};
        double decode_seconds{};
        std::size_t prefill_matrix_passes{};
        std::size_t runtime_tensor_lookups{};
        std::size_t decode_parallel_dispatches{};
        std::uint64_t final_logits_hash{};
        std::uint64_t final_kv_hash{};
        std::vector<std::size_t> final_kv_lengths;
        std::vector<std::uint64_t> frontier_logits_hashes;
        std::vector<std::uint64_t> frontier_kv_hashes;
    };

    struct TokenWaveGeneration {
        Generation generation;
        std::vector<std::size_t> acceptance_lengths;
        std::vector<double> traversal_seconds;
        std::size_t target_weight_traversals{};
        std::size_t committed_decode_tokens{};
        std::size_t rolled_back_tokens{};
        std::size_t speculative_traversals{};
        std::size_t fallback_traversals{};
        std::size_t transactional_snapshot_bytes_copied{};
        bool rollback_verified{true};
    };

    struct BatchGeneration {
        std::vector<std::vector<int>> tokens;
        double prefill_seconds{};
        double decode_seconds{};
        std::size_t decode_matrix_passes{};
        std::size_t decode_parallel_dispatches{};
    };

    Generation generate(
        const std::vector<int>& prompt,
        std::size_t max_new_tokens
    ) {
        if (prompt.empty()) throw std::runtime_error("prompt cannot be empty");
        if (prompt.size() + max_new_tokens > descriptor_.config.max_position_embeddings) {
            throw std::runtime_error("generation exceeds model context limit");
        }
        const auto prefill_start = Clock::now();
        matrix_passes_ = 0;
        std::vector<float> logits = forward_prompt(prompt, caches_);
        const auto prefill_end = Clock::now();

        Generation generation;
        generation.prefill_seconds = seconds(prefill_start, prefill_end);
        generation.prefill_matrix_passes = matrix_passes_;
        parallel_.reset_dispatches();
        const auto decode_start = Clock::now();
        for (std::size_t index = 0; index < max_new_tokens; ++index) {
            generation.frontier_logits_hashes.push_back(hash_floats(logits));
            generation.frontier_kv_hashes.push_back(hash_caches(caches_));
            const int token = static_cast<int>(
                std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()))
            );
            generation.tokens.push_back(token);
            if (token == descriptor_.config.eos_token_id || index + 1 == max_new_tokens) {
                break;
            }
            const auto token_start = Clock::now();
            logits = forward_token(token, true, caches_);
            generation.inter_token_seconds.push_back(seconds(token_start, Clock::now()));
        }
        generation.decode_seconds = seconds(decode_start, Clock::now());
        generation.runtime_tensor_lookups = tensor_lookups_;
        generation.decode_parallel_dispatches = parallel_.dispatches();
        generation.final_logits_hash = hash_floats(logits);
        generation.final_kv_hash = hash_caches(caches_);
        generation.final_kv_lengths = cache_lengths(caches_);
        return generation;
    }

    TokenWaveGeneration generate_token_wave(
        const std::vector<int>& prompt,
        std::size_t max_new_tokens,
        std::size_t max_proposals,
        bool adversarial_proposals
    ) {
        if (prompt.empty()) throw std::runtime_error("prompt cannot be empty");
        if (prompt.size() + max_new_tokens > descriptor_.config.max_position_embeddings) {
            throw std::runtime_error("token wave exceeds model context limit");
        }
        std::vector<LayerCache> states;
        initialize_caches(states);
        matrix_passes_ = 0;
        const auto prefill_start = Clock::now();
        std::vector<float> logits = forward_prompt(prompt, states);
        const auto prefill_end = Clock::now();
        TokenWaveGeneration wave;
        wave.generation.prefill_seconds = seconds(prefill_start, prefill_end);
        wave.generation.prefill_matrix_passes = matrix_passes_;
        std::vector<int> history = prompt;
        std::size_t speculation_backoff = 0;
        parallel_.reset_dispatches();
        const auto decode_start = Clock::now();
        while (wave.generation.tokens.size() < max_new_tokens) {
            wave.generation.frontier_logits_hashes.push_back(hash_floats(logits));
            wave.generation.frontier_kv_hashes.push_back(hash_caches(states));
            const int pending = greedy_token(logits);
            wave.generation.tokens.push_back(pending);
            history.push_back(pending);
            if (pending == descriptor_.config.eos_token_id ||
                wave.generation.tokens.size() == max_new_tokens) {
                break;
            }
            const std::size_t remaining = max_new_tokens - wave.generation.tokens.size();
            const std::size_t proposal_limit = remaining > 1
                ? std::min(max_proposals, remaining - 1)
                : 0;
            auto proposals = speculation_backoff == 0
                ? propose_ngrams(history, proposal_limit)
                : std::vector<int>{};
            if (speculation_backoff > 0) --speculation_backoff;
            if (adversarial_proposals && !proposals.empty()) {
                proposals[0] = (proposals[0] + 1) % descriptor_.config.vocab_size;
            }
            if (proposals.empty()) {
                const auto traversal_start = Clock::now();
                logits = forward_token(pending, true, states);
                wave.traversal_seconds.push_back(seconds(traversal_start, Clock::now()));
                ++wave.target_weight_traversals;
                ++wave.committed_decode_tokens;
                ++wave.fallback_traversals;
                wave.acceptance_lengths.push_back(0);
                continue;
            }
            std::vector<int> block{pending};
            block.insert(block.end(), proposals.begin(), proposals.end());
            const auto before = snapshot_caches(states);
            const auto traversal_start = Clock::now();
            auto block_logits = forward_block_logits(block, states);
            wave.traversal_seconds.push_back(seconds(traversal_start, Clock::now()));
            ++wave.target_weight_traversals;
            ++wave.speculative_traversals;
            std::size_t accepted = 0;
            while (accepted < proposals.size()) {
                const auto begin = block_logits.begin() + static_cast<std::ptrdiff_t>(
                    accepted * descriptor_.config.vocab_size
                );
                std::vector<float> position_logits(
                    begin,
                    begin + static_cast<std::ptrdiff_t>(descriptor_.config.vocab_size)
                );
                if (greedy_token(position_logits) != proposals[accepted]) break;
                auto committed_state = states;
                rollback_caches(committed_state, before, 1 + accepted);
                wave.generation.frontier_logits_hashes.push_back(
                    hash_floats(position_logits)
                );
                wave.generation.frontier_kv_hashes.push_back(
                    hash_caches(committed_state)
                );
                wave.generation.tokens.push_back(proposals[accepted]);
                history.push_back(proposals[accepted]);
                ++accepted;
            }
            wave.acceptance_lengths.push_back(accepted);
            if (accepted == 0) speculation_backoff = 2;
            const std::size_t committed = 1 + accepted;
            wave.committed_decode_tokens += committed;
            if (committed < block.size()) {
                wave.rolled_back_tokens += block.size() - committed;
                rollback_caches(states, before, committed);
                wave.rollback_verified = wave.rollback_verified &&
                    verify_rollback(states, before, committed);
            }
            const auto next_begin = block_logits.begin() + static_cast<std::ptrdiff_t>(
                accepted * descriptor_.config.vocab_size
            );
            logits.assign(
                next_begin,
                next_begin + static_cast<std::ptrdiff_t>(descriptor_.config.vocab_size)
            );
        }
        wave.generation.decode_seconds = seconds(decode_start, Clock::now());
        wave.generation.runtime_tensor_lookups = tensor_lookups_;
        wave.generation.decode_parallel_dispatches = parallel_.dispatches();
        wave.generation.final_logits_hash = hash_floats(logits);
        wave.generation.final_kv_hash = hash_caches(states);
        wave.generation.final_kv_lengths = cache_lengths(states);
        return wave;
    }

    BatchGeneration generate_batch(
        const std::vector<std::vector<int>>& prompts,
        std::size_t max_new_tokens
    ) {
        if (prompts.empty()) throw std::runtime_error("batch cannot be empty");
        const auto& config = descriptor_.config;
        for (const auto& prompt : prompts) {
            if (prompt.empty()) throw std::runtime_error("batch prompt cannot be empty");
            if (prompt.size() + max_new_tokens > config.max_position_embeddings) {
                throw std::runtime_error("batch generation exceeds model context limit");
            }
        }

        std::vector<std::vector<LayerCache>> states(prompts.size());
        for (auto& state : states) initialize_caches(state);
        std::vector<std::vector<float>> logits(prompts.size());
        const auto prefill_start = Clock::now();
        for (std::size_t request = 0; request < prompts.size(); ++request) {
            logits[request] = forward_prompt(prompts[request], states[request]);
        }
        const auto prefill_end = Clock::now();

        BatchGeneration generation;
        generation.tokens.resize(prompts.size());
        generation.prefill_seconds = seconds(prefill_start, prefill_end);
        matrix_passes_ = 0;
        parallel_.reset_dispatches();
        const auto decode_start = Clock::now();
        for (std::size_t step = 0; step < max_new_tokens; ++step) {
            std::vector<int> pending_tokens;
            std::vector<std::size_t> pending_requests;
            std::vector<std::vector<LayerCache>*> pending_states;
            for (std::size_t request = 0; request < prompts.size(); ++request) {
                if (logits[request].empty()) continue;
                const int token = static_cast<int>(std::distance(
                    logits[request].begin(),
                    std::max_element(logits[request].begin(), logits[request].end())
                ));
                generation.tokens[request].push_back(token);
                if (token == config.eos_token_id || step + 1 == max_new_tokens) {
                    logits[request].clear();
                    continue;
                }
                pending_tokens.push_back(token);
                pending_requests.push_back(request);
                pending_states.push_back(&states[request]);
            }
            if (pending_tokens.empty()) break;
            auto next_logits = forward_batch_tokens(pending_tokens, pending_states);
            for (std::size_t index = 0; index < pending_requests.size(); ++index) {
                logits[pending_requests[index]] = std::move(next_logits[index]);
            }
        }
        generation.decode_seconds = seconds(decode_start, Clock::now());
        generation.decode_matrix_passes = matrix_passes_;
        generation.decode_parallel_dispatches = parallel_.dispatches();
        return generation;
    }

    std::string weight_dtype() const {
        return embedding_weight_->dtype == DType::BF16 ? "BF16" : "F32";
    }

    const char* kernel_family() const { return kernels_.name; }

    std::size_t threads() const { return threads_; }

    std::size_t rope_table_entries() const { return rope_cos_.size(); }

    std::size_t attention_workspace_floats() const { return attention_scores_.size(); }

    std::size_t activation_panel_width() const { return activation_panel_width_; }

private:
    using Clock = std::chrono::steady_clock;

    void validate_tensor_ranges() const {
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        ranges.reserve(descriptor_.tensors.size());
        for (const auto& [name, tensor] : descriptor_.tensors) {
            if (tensor.offset > weights_.size() ||
                tensor.byte_size > weights_.size() - tensor.offset) {
                throw std::runtime_error("tensor range outside checkpoint: " + name);
            }
            ranges.emplace_back(tensor.offset, tensor.offset + tensor.byte_size);
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t index = 1; index < ranges.size(); ++index) {
            if (ranges[index].first < ranges[index - 1].second) {
                throw std::runtime_error("overlapping tensor ranges in descriptor");
            }
        }
    }

    struct LayerCache {
        std::vector<float> key;
        std::vector<float> value;
        std::size_t length{};
    };

    struct CacheMark {
        std::size_t key_size{};
        std::size_t value_size{};
        std::size_t length{};
        std::uint64_t key_hash{};
        std::uint64_t value_hash{};
    };

    using CacheSnapshot = std::vector<CacheMark>;

    static int greedy_token(const std::vector<float>& logits) {
        return static_cast<int>(std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end())
        ));
    }

    static std::uint64_t hash_bytes(
        std::uint64_t hash, const void* data, std::size_t size
    ) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    static std::uint64_t hash_floats(const std::vector<float>& values) {
        return hash_bytes(
            1469598103934665603ULL, values.data(), values.size() * sizeof(float)
        );
    }

    static std::uint64_t hash_caches(const std::vector<LayerCache>& caches) {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const auto& cache : caches) {
            hash = hash_bytes(hash, &cache.length, sizeof(cache.length));
            hash = hash_bytes(hash, cache.key.data(), cache.key.size() * sizeof(float));
            hash = hash_bytes(hash, cache.value.data(), cache.value.size() * sizeof(float));
        }
        return hash;
    }

    static std::vector<std::size_t> cache_lengths(
        const std::vector<LayerCache>& caches
    ) {
        std::vector<std::size_t> lengths;
        for (const auto& cache : caches) lengths.push_back(cache.length);
        return lengths;
    }

    static CacheSnapshot snapshot_caches(const std::vector<LayerCache>& caches) {
        CacheSnapshot snapshot;
        snapshot.reserve(caches.size());
        for (const auto& cache : caches) {
            snapshot.push_back(CacheMark{
                cache.key.size(),
                cache.value.size(),
                cache.length,
                hash_floats(cache.key),
                hash_floats(cache.value),
            });
        }
        return snapshot;
    }

    std::size_t key_value_width() const {
        return descriptor_.config.num_key_value_heads * head_dim_;
    }

    void rollback_caches(
        std::vector<LayerCache>& caches,
        const CacheSnapshot& before,
        std::size_t committed
    ) const {
        const std::size_t added = committed * key_value_width();
        for (std::size_t layer = 0; layer < caches.size(); ++layer) {
            caches[layer].key.resize(before[layer].key_size + added);
            caches[layer].value.resize(before[layer].value_size + added);
            caches[layer].length = before[layer].length + committed;
        }
    }

    bool verify_rollback(
        const std::vector<LayerCache>& caches,
        const CacheSnapshot& before,
        std::size_t committed
    ) const {
        const std::size_t added = committed * key_value_width();
        for (std::size_t layer = 0; layer < caches.size(); ++layer) {
            const auto& current = caches[layer];
            const auto& original = before[layer];
            if (current.length != original.length + committed ||
                current.key.size() != original.key_size + added ||
                current.value.size() != original.value_size + added ||
                hash_bytes(
                    1469598103934665603ULL,
                    current.key.data(),
                    original.key_size * sizeof(float)
                ) != original.key_hash ||
                hash_bytes(
                    1469598103934665603ULL,
                    current.value.data(),
                    original.value_size * sizeof(float)
                ) != original.value_hash) {
                return false;
            }
        }
        return true;
    }

    static std::vector<int> propose_ngrams(
        const std::vector<int>& history, std::size_t limit
    ) {
        if (limit == 0 || history.size() < 2) return {};
        const std::size_t maximum_order = std::min<std::size_t>(4, history.size() - 1);
        for (std::size_t order = maximum_order; order >= 2; --order) {
            const std::size_t suffix = history.size() - order;
            std::vector<int> supported;
            std::size_t support = 0;
            for (std::size_t candidate = suffix; candidate-- > 0;) {
                if (candidate + order >= history.size()) continue;
                if (!std::equal(
                        history.begin() + static_cast<std::ptrdiff_t>(candidate),
                        history.begin() + static_cast<std::ptrdiff_t>(candidate + order),
                        history.begin() + static_cast<std::ptrdiff_t>(suffix))) {
                    continue;
                }
                const std::size_t continuation = candidate + order;
                const std::size_t count = std::min(limit, history.size() - continuation);
                if (count < 2) continue;
                std::vector<int> continuation_tokens(
                    history.begin() + static_cast<std::ptrdiff_t>(continuation),
                    history.begin() + static_cast<std::ptrdiff_t>(continuation + count)
                );
                if (supported.empty()) {
                    supported = std::move(continuation_tokens);
                    support = 1;
                } else if (continuation_tokens == supported) {
                    ++support;
                }
            }
            if (support >= 2) return supported;
            if (order >= 4 && support == 1) {
                std::size_t continuation_support = 0;
                for (std::size_t start = 0;
                     start + supported.size() <= history.size(); ++start) {
                    if (std::equal(
                            supported.begin(), supported.end(),
                            history.begin() + static_cast<std::ptrdiff_t>(start))) {
                        ++continuation_support;
                    }
                }
                if (continuation_support >= 2) return supported;
            }
        }
        return {};
    }

    void initialize_caches(std::vector<LayerCache>& caches) const {
        const auto& config = descriptor_.config;
        caches.resize(config.num_hidden_layers);
        const std::size_t capacity =
            config.max_position_embeddings * config.num_key_value_heads * head_dim_;
        for (auto& cache : caches) {
            cache.key.reserve(capacity);
            cache.value.reserve(capacity);
        }
    }

    static double seconds(Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration<double>(end - start).count();
    }

    const Tensor& tensor(const std::string& name) const {
        ++tensor_lookups_;
        const auto found = descriptor_.tensors.find(name);
        if (found == descriptor_.tensors.end()) {
            throw std::runtime_error("checkpoint is missing tensor " + name);
        }
        return found->second;
    }

    float tensor_value(const Tensor& value, std::size_t index) const {
        if (value.dtype == DType::BF16) {
            const auto* data = reinterpret_cast<const std::uint16_t*>(weights_.at(value.offset));
            return bf16_to_float(data[index]);
        }
        const auto* data = reinterpret_cast<const float*>(weights_.at(value.offset));
        return data[index];
    }

    float quantize(float value, DType dtype) const {
        return dtype == DType::BF16 ? round_bf16(value) : value;
    }

    std::vector<float> embedding(int token_id) const {
        const auto& weight = *embedding_weight_;
        if (weight.shape.size() != 2 || token_id < 0 ||
            static_cast<std::size_t>(token_id) >= weight.shape[0]) {
            throw std::runtime_error("token outside embedding table");
        }
        std::vector<float> output(weight.shape[1]);
        const std::size_t start = static_cast<std::size_t>(token_id) * weight.shape[1];
        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] = tensor_value(weight, start + index);
        }
        return output;
    }

    void linear_rows(
        const std::vector<float>& input,
        const Tensor& weight,
        std::vector<float>& output,
        std::size_t begin,
        std::size_t end
    ) const {
        const std::size_t columns = weight.shape[1];
        if (weight.dtype == DType::BF16) {
            const auto* data = reinterpret_cast<const std::uint16_t*>(weights_.at(weight.offset));
            kernels_.bf16_rows(data, input.data(), output.data(), columns, begin, end);
        } else {
            const auto* data = reinterpret_cast<const float*>(weights_.at(weight.offset));
            kernels_.f32_rows(data, input.data(), output.data(), columns, begin, end);
        }
    }

    void validate_linear(const std::vector<float>& input, const Tensor& weight) const {
        if (weight.shape.size() != 2 || weight.shape[1] != input.size()) {
            throw std::runtime_error("native linear shape mismatch");
        }
    }

    std::vector<float> linear(const std::vector<float>& input, const Tensor& weight) {
        ++matrix_passes_;
        validate_linear(input, weight);
        const std::size_t rows = weight.shape[0];
        std::vector<float> output(rows);
        parallel_.run(rows, 16, [&](std::size_t begin, std::size_t end) {
            linear_rows(input, weight, output, begin, end);
        });
        return output;
    }

    void linear_pair(
        const std::vector<float>& input,
        const Tensor& first,
        const Tensor& second,
        std::vector<float>& first_output,
        std::vector<float>& second_output
    ) {
        matrix_passes_ += 2;
        validate_linear(input, first);
        validate_linear(input, second);
        const std::size_t first_rows = first.shape[0];
        const std::size_t second_rows = second.shape[0];
        first_output.resize(first_rows);
        second_output.resize(second_rows);
        parallel_.run(first_rows + second_rows, 64, [&](std::size_t begin, std::size_t end) {
            const std::size_t first_begin = std::min(begin, first_rows);
            const std::size_t first_end = std::min(end, first_rows);
            if (first_begin < first_end) {
                linear_rows(input, first, first_output, first_begin, first_end);
            }
            const std::size_t second_begin = begin > first_rows ? begin - first_rows : 0;
            const std::size_t second_end = end > first_rows
                ? std::min(end - first_rows, second_rows)
                : 0;
            if (second_begin < second_end) {
                linear_rows(input, second, second_output, second_begin, second_end);
            }
        });
    }

    void linear_triple(
        const std::vector<float>& input,
        const Tensor& first,
        const Tensor& second,
        const Tensor& third,
        std::vector<float>& first_output,
        std::vector<float>& second_output,
        std::vector<float>& third_output
    ) {
        matrix_passes_ += 3;
        validate_linear(input, first);
        validate_linear(input, second);
        validate_linear(input, third);
        const std::size_t first_rows = first.shape[0];
        const std::size_t second_rows = second.shape[0];
        const std::size_t third_rows = third.shape[0];
        const std::size_t second_offset = first_rows;
        const std::size_t third_offset = first_rows + second_rows;
        first_output.resize(first_rows);
        second_output.resize(second_rows);
        third_output.resize(third_rows);
        parallel_.run(third_offset + third_rows, 64, [&](std::size_t begin, std::size_t end) {
            const std::size_t first_begin = std::min(begin, first_rows);
            const std::size_t first_end = std::min(end, first_rows);
            if (first_begin < first_end) {
                linear_rows(input, first, first_output, first_begin, first_end);
            }
            const std::size_t second_begin = begin > second_offset ? begin - second_offset : 0;
            const std::size_t second_end = end > second_offset
                ? std::min(end - second_offset, second_rows)
                : 0;
            if (second_begin < second_end) {
                linear_rows(input, second, second_output, second_begin, second_end);
            }
            const std::size_t third_begin = begin > third_offset ? begin - third_offset : 0;
            const std::size_t third_end = end > third_offset
                ? std::min(end - third_offset, third_rows)
                : 0;
            if (third_begin < third_end) {
                linear_rows(input, third, third_output, third_begin, third_end);
            }
        });
    }

    static constexpr std::size_t activation_panel_width_ = 16;

    static std::size_t activation_panel_stride(std::size_t positions) {
        return ((positions + activation_panel_width_ - 1) / activation_panel_width_) *
               activation_panel_width_;
    }

    static std::vector<float> pack_activation_panel(
        const std::vector<float>& input,
        std::size_t positions,
        std::size_t columns,
        std::size_t stride
    ) {
        std::vector<float> panel(columns * stride, 0.0F);
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t position = 0; position < positions; ++position) {
                panel[column * stride + position] = input[position * columns + column];
            }
        }
        return panel;
    }

    std::vector<float> linear_block(
        const std::vector<float>& input,
        std::size_t positions,
        const Tensor& weight
    ) {
        ++matrix_passes_;
        validate_linear_block(input, positions, weight);
        const std::size_t rows = weight.shape[0];
        const std::size_t columns = weight.shape[1];
        const std::size_t stride = activation_panel_stride(positions);
        const auto panel = pack_activation_panel(input, positions, columns, stride);
        std::vector<float> output(positions * rows);
        parallel_.run(rows, 16, [&](std::size_t begin, std::size_t end) {
            linear_block_rows(
                panel, positions, stride, weight, output, begin, end
            );
        });
        return output;
    }

    void linear_block_rows(
        const std::vector<float>& panel,
        std::size_t positions,
        std::size_t panel_stride,
        const Tensor& weight,
        std::vector<float>& output,
        std::size_t begin,
        std::size_t end
    ) const {
        const std::size_t rows = weight.shape[0];
        const std::size_t columns = weight.shape[1];
        if (weight.dtype == DType::BF16) {
            const auto* data = reinterpret_cast<const std::uint16_t*>(
                weights_.at(weight.offset)
            );
            kernels_.bf16_panel(
                data,
                panel.data(),
                output.data(),
                rows,
                columns,
                positions,
                panel_stride,
                begin,
                end
            );
        } else {
            const auto* data = reinterpret_cast<const float*>(weights_.at(weight.offset));
            kernels_.f32_panel(
                data,
                panel.data(),
                output.data(),
                rows,
                columns,
                positions,
                panel_stride,
                begin,
                end
            );
        }
    }

    void validate_linear_block(
        const std::vector<float>& input,
        std::size_t positions,
        const Tensor& weight
    ) const {
        if (weight.shape.size() != 2 || positions == 0 ||
            input.size() != positions * weight.shape[1]) {
            throw std::runtime_error("native block linear shape mismatch");
        }
    }

    void linear_block_pair(
        const std::vector<float>& input,
        std::size_t positions,
        const Tensor& first,
        const Tensor& second,
        std::vector<float>& first_output,
        std::vector<float>& second_output
    ) {
        matrix_passes_ += 2;
        validate_linear_block(input, positions, first);
        validate_linear_block(input, positions, second);
        const std::size_t first_rows = first.shape[0];
        const std::size_t second_rows = second.shape[0];
        const std::size_t stride = activation_panel_stride(positions);
        const auto panel = pack_activation_panel(
            input, positions, first.shape[1], stride
        );
        first_output.resize(positions * first_rows);
        second_output.resize(positions * second_rows);
        parallel_.run(first_rows + second_rows, 64, [&](std::size_t begin, std::size_t end) {
            const std::size_t first_begin = std::min(begin, first_rows);
            const std::size_t first_end = std::min(end, first_rows);
            if (first_begin < first_end) {
                linear_block_rows(
                    panel, positions, stride, first, first_output, first_begin, first_end
                );
            }
            const std::size_t second_begin = begin > first_rows ? begin - first_rows : 0;
            const std::size_t second_end = end > first_rows
                ? std::min(end - first_rows, second_rows)
                : 0;
            if (second_begin < second_end) {
                linear_block_rows(
                    panel,
                    positions,
                    stride,
                    second,
                    second_output,
                    second_begin,
                    second_end
                );
            }
        });
    }

    void linear_block_triple(
        const std::vector<float>& input,
        std::size_t positions,
        const Tensor& first,
        const Tensor& second,
        const Tensor& third,
        std::vector<float>& first_output,
        std::vector<float>& second_output,
        std::vector<float>& third_output
    ) {
        matrix_passes_ += 3;
        validate_linear_block(input, positions, first);
        validate_linear_block(input, positions, second);
        validate_linear_block(input, positions, third);
        const std::size_t first_rows = first.shape[0];
        const std::size_t second_rows = second.shape[0];
        const std::size_t third_rows = third.shape[0];
        const std::size_t second_offset = first_rows;
        const std::size_t third_offset = first_rows + second_rows;
        const std::size_t stride = activation_panel_stride(positions);
        const auto panel = pack_activation_panel(
            input, positions, first.shape[1], stride
        );
        first_output.resize(positions * first_rows);
        second_output.resize(positions * second_rows);
        third_output.resize(positions * third_rows);
        parallel_.run(third_offset + third_rows, 64, [&](std::size_t begin, std::size_t end) {
            const std::size_t first_begin = std::min(begin, first_rows);
            const std::size_t first_end = std::min(end, first_rows);
            if (first_begin < first_end) {
                linear_block_rows(
                    panel, positions, stride, first, first_output, first_begin, first_end
                );
            }
            const std::size_t second_begin = begin > second_offset ? begin - second_offset : 0;
            const std::size_t second_end = end > second_offset
                ? std::min(end - second_offset, second_rows)
                : 0;
            if (second_begin < second_end) {
                linear_block_rows(
                    panel,
                    positions,
                    stride,
                    second,
                    second_output,
                    second_begin,
                    second_end
                );
            }
            const std::size_t third_begin = begin > third_offset ? begin - third_offset : 0;
            const std::size_t third_end = end > third_offset
                ? std::min(end - third_offset, third_rows)
                : 0;
            if (third_begin < third_end) {
                linear_block_rows(
                    panel, positions, stride, third, third_output, third_begin, third_end
                );
            }
        });
    }

    std::vector<float> rms_norm(
        const std::vector<float>& input,
        const Tensor& weight
    ) const {
        if (weight.shape.size() != 1 || weight.shape[0] != input.size()) {
            throw std::runtime_error("native RMSNorm shape mismatch");
        }
        float squares = 0.0F;
        for (float value : input) squares += value * value;
        const float inverse = 1.0F / std::sqrt(
            squares / static_cast<float>(input.size()) + descriptor_.config.rms_norm_eps
        );
        std::vector<float> output(input.size());
        for (std::size_t index = 0; index < input.size(); ++index) {
            const float normalized = quantize(input[index] * inverse, weight.dtype);
            output[index] = quantize(normalized * tensor_value(weight, index), weight.dtype);
        }
        return output;
    }

    std::vector<float> rms_norm_block(
        const std::vector<float>& input,
        std::size_t positions,
        const Tensor& weight
    ) const {
        const std::size_t width = weight.shape.empty() ? 0 : weight.shape[0];
        if (weight.shape.size() != 1 || positions == 0 ||
            input.size() != positions * width) {
            throw std::runtime_error("native block RMSNorm shape mismatch");
        }
        std::vector<float> output(input.size());
        for (std::size_t position = 0; position < positions; ++position) {
            const std::size_t base = position * width;
            float squares = 0.0F;
            for (std::size_t index = 0; index < width; ++index) {
                const float value = input[base + index];
                squares += value * value;
            }
            const float inverse = 1.0F / std::sqrt(
                squares / static_cast<float>(width) + descriptor_.config.rms_norm_eps
            );
            for (std::size_t index = 0; index < width; ++index) {
                const float normalized = quantize(input[base + index] * inverse, weight.dtype);
                output[base + index] = quantize(
                    normalized * tensor_value(weight, index), weight.dtype
                );
            }
        }
        return output;
    }

    void apply_rope(
        std::vector<float>& query,
        std::vector<float>& key,
        std::size_t position,
        DType dtype
    ) const {
        const auto& config = descriptor_.config;
        const std::size_t half = head_dim_ / 2;
        for (std::size_t head = 0; head < config.num_attention_heads; ++head) {
            const std::size_t base = head * head_dim_;
            for (std::size_t index = 0; index < half; ++index) {
                const std::size_t table_index = position * half + index;
                const float cosine = rope_cos_[table_index];
                const float sine = rope_sin_[table_index];
                const float first = query[base + index];
                const float second = query[base + half + index];
                query[base + index] = quantize(
                    quantize(first * cosine, dtype) - quantize(second * sine, dtype), dtype
                );
                query[base + half + index] = quantize(
                    quantize(second * cosine, dtype) + quantize(first * sine, dtype), dtype
                );
            }
        }
        for (std::size_t head = 0; head < config.num_key_value_heads; ++head) {
            const std::size_t base = head * head_dim_;
            for (std::size_t index = 0; index < half; ++index) {
                const std::size_t table_index = position * half + index;
                const float cosine = rope_cos_[table_index];
                const float sine = rope_sin_[table_index];
                const float first = key[base + index];
                const float second = key[base + half + index];
                key[base + index] = quantize(
                    quantize(first * cosine, dtype) - quantize(second * sine, dtype), dtype
                );
                key[base + half + index] = quantize(
                    quantize(second * cosine, dtype) + quantize(first * sine, dtype), dtype
                );
            }
        }
    }

    std::vector<float> attention(
        const std::vector<float>& query,
        LayerCache& cache,
        DType dtype,
        std::size_t visible_length
    ) const {
        const auto& config = descriptor_.config;
        const std::size_t repeats = config.num_attention_heads / config.num_key_value_heads;
        std::vector<float> output(config.hidden_size);
        for (std::size_t head = 0; head < config.num_attention_heads; ++head) {
            const std::size_t kv_head = head / repeats;
            auto& scores = attention_scores_;
            float maximum = -INFINITY;
            for (std::size_t token = 0; token < visible_length; ++token) {
                float score = 0.0F;
                const std::size_t query_base = head * head_dim_;
                const std::size_t key_base =
                    (token * config.num_key_value_heads + kv_head) * head_dim_;
                for (std::size_t dimension = 0; dimension < head_dim_; ++dimension) {
                    score += query[query_base + dimension] *
                             cache.key[key_base + dimension];
                }
                score = quantize(score, dtype);
                score = quantize(score / std::sqrt(static_cast<float>(head_dim_)), dtype);
                scores[token] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0F;
            for (std::size_t token = 0; token < visible_length; ++token) {
                scores[token] = std::exp(scores[token] - maximum);
                denominator += scores[token];
            }
            for (std::size_t token = 0; token < visible_length; ++token) {
                scores[token] = quantize(scores[token] / denominator, dtype);
            }
            for (std::size_t dimension = 0; dimension < head_dim_; ++dimension) {
                float sum = 0.0F;
                for (std::size_t token = 0; token < visible_length; ++token) {
                    const std::size_t value_index =
                        (token * config.num_key_value_heads + kv_head) * head_dim_ + dimension;
                    sum += scores[token] * cache.value[value_index];
                }
                output[head * head_dim_ + dimension] = quantize(sum, dtype);
            }
        }
        return output;
    }

    std::vector<float> forward_prompt(
        const std::vector<int>& prompt,
        std::vector<LayerCache>& caches,
        bool return_all_logits = false
    ) {
        const auto& config = descriptor_.config;
        const std::size_t positions = prompt.size();
        const std::size_t hidden_width = config.hidden_size;
        const std::size_t key_value_width = config.num_key_value_heads * head_dim_;
        std::vector<float> hidden(positions * hidden_width);
        for (std::size_t position = 0; position < positions; ++position) {
            const auto token_embedding = embedding(prompt[position]);
            std::copy(
                token_embedding.begin(),
                token_embedding.end(),
                hidden.begin() + static_cast<std::ptrdiff_t>(position * hidden_width)
            );
        }
        const DType dtype = embedding_weight_->dtype;
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const auto& plan = layers_[layer];
            auto residual = hidden;
            auto normalized = rms_norm_block(
                hidden, positions, *plan.input_norm
            );
            auto query = linear_block(
                normalized, positions, *plan.query
            );
            auto key = linear_block(
                normalized, positions, *plan.key
            );
            auto value = linear_block(
                normalized, positions, *plan.value
            );
            auto& cache = caches[layer];
            const std::size_t previous_length = cache.length;
            for (std::size_t position = 0; position < positions; ++position) {
                const auto query_begin = query.begin() +
                    static_cast<std::ptrdiff_t>(position * hidden_width);
                const auto key_begin = key.begin() +
                    static_cast<std::ptrdiff_t>(position * key_value_width);
                std::vector<float> position_query(query_begin, query_begin + hidden_width);
                std::vector<float> position_key(key_begin, key_begin + key_value_width);
                apply_rope(
                    position_query,
                    position_key,
                    previous_length + position,
                    dtype
                );
                std::copy(position_query.begin(), position_query.end(), query_begin);
                std::copy(position_key.begin(), position_key.end(), key_begin);
            }
            cache.key.insert(cache.key.end(), key.begin(), key.end());
            cache.value.insert(cache.value.end(), value.begin(), value.end());
            cache.length += positions;

            std::vector<float> attended(positions * hidden_width);
            for (std::size_t position = 0; position < positions; ++position) {
                const auto query_begin = query.begin() +
                    static_cast<std::ptrdiff_t>(position * hidden_width);
                std::vector<float> position_query(query_begin, query_begin + hidden_width);
                auto position_attended = attention(
                    position_query,
                    cache,
                    dtype,
                    previous_length + position + 1
                );
                std::copy(
                    position_attended.begin(),
                    position_attended.end(),
                    attended.begin() +
                        static_cast<std::ptrdiff_t>(position * hidden_width)
                );
            }
            auto projected = linear_block(
                attended, positions, *plan.attention_output
            );
            for (std::size_t index = 0; index < hidden.size(); ++index) {
                hidden[index] = quantize(residual[index] + projected[index], dtype);
            }

            residual = hidden;
            normalized = rms_norm_block(
                hidden,
                positions,
                *plan.post_attention_norm
            );
            auto gate = linear_block(
                normalized, positions, *plan.gate
            );
            auto up = linear_block(
                normalized, positions, *plan.up
            );
            for (std::size_t index = 0; index < gate.size(); ++index) {
                const float silu = quantize(
                    gate[index] / (1.0F + std::exp(-gate[index])), dtype
                );
                gate[index] = quantize(silu * up[index], dtype);
            }
            auto down = linear_block(
                gate, positions, *plan.down
            );
            for (std::size_t index = 0; index < hidden.size(); ++index) {
                hidden[index] = quantize(residual[index] + down[index], dtype);
            }
        }

        hidden = rms_norm_block(hidden, positions, *final_norm_);
        auto all_logits = linear_block(hidden, positions, *lm_head_);
        if (return_all_logits) return all_logits;
        const auto final_begin = all_logits.end() -
            static_cast<std::ptrdiff_t>(config.vocab_size);
        return std::vector<float>(final_begin, all_logits.end());
    }

    std::vector<float> forward_block_logits(
        const std::vector<int>& tokens,
        std::vector<LayerCache>& caches
    ) {
        return forward_prompt(tokens, caches, true);
    }

    std::vector<std::vector<float>> forward_batch_tokens(
        const std::vector<int>& token_ids,
        const std::vector<std::vector<LayerCache>*>& states
    ) {
        if (token_ids.empty() || token_ids.size() != states.size()) {
            throw std::runtime_error("native batch token/state mismatch");
        }
        const auto& config = descriptor_.config;
        const std::size_t batch = token_ids.size();
        const std::size_t hidden_width = config.hidden_size;
        const std::size_t key_value_width = config.num_key_value_heads * head_dim_;
        std::vector<float> hidden(batch * hidden_width);
        for (std::size_t request = 0; request < batch; ++request) {
            const auto token_embedding = embedding(token_ids[request]);
            std::copy(
                token_embedding.begin(),
                token_embedding.end(),
                hidden.begin() + static_cast<std::ptrdiff_t>(request * hidden_width)
            );
        }
        const DType dtype = embedding_weight_->dtype;
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const auto& plan = layers_[layer];
            auto residual = hidden;
            auto normalized = rms_norm_block(hidden, batch, *plan.input_norm);
            std::vector<float> query;
            std::vector<float> key;
            std::vector<float> value;
            linear_block_triple(
                normalized,
                batch,
                *plan.query,
                *plan.key,
                *plan.value,
                query,
                key,
                value
            );

            std::vector<float> attended(batch * hidden_width);
            for (std::size_t request = 0; request < batch; ++request) {
                const auto query_begin = query.begin() +
                    static_cast<std::ptrdiff_t>(request * hidden_width);
                const auto key_begin = key.begin() +
                    static_cast<std::ptrdiff_t>(request * key_value_width);
                const auto value_begin = value.begin() +
                    static_cast<std::ptrdiff_t>(request * key_value_width);
                std::vector<float> request_query(
                    query_begin, query_begin + static_cast<std::ptrdiff_t>(hidden_width)
                );
                std::vector<float> request_key(
                    key_begin, key_begin + static_cast<std::ptrdiff_t>(key_value_width)
                );
                auto& cache = (*states[request])[layer];
                apply_rope(request_query, request_key, cache.length, dtype);
                cache.key.insert(cache.key.end(), request_key.begin(), request_key.end());
                cache.value.insert(
                    cache.value.end(),
                    value_begin,
                    value_begin + static_cast<std::ptrdiff_t>(key_value_width)
                );
                ++cache.length;
                const auto request_attended = attention(
                    request_query, cache, dtype, cache.length
                );
                std::copy(
                    request_attended.begin(),
                    request_attended.end(),
                    attended.begin() +
                        static_cast<std::ptrdiff_t>(request * hidden_width)
                );
            }

            auto projected = linear_block(
                attended, batch, *plan.attention_output
            );
            for (std::size_t index = 0; index < hidden.size(); ++index) {
                hidden[index] = quantize(residual[index] + projected[index], dtype);
            }

            residual = hidden;
            normalized = rms_norm_block(hidden, batch, *plan.post_attention_norm);
            std::vector<float> gate;
            std::vector<float> up;
            linear_block_pair(
                normalized, batch, *plan.gate, *plan.up, gate, up
            );
            for (std::size_t index = 0; index < gate.size(); ++index) {
                const float silu = quantize(
                    gate[index] / (1.0F + std::exp(-gate[index])), dtype
                );
                gate[index] = quantize(silu * up[index], dtype);
            }
            auto down = linear_block(gate, batch, *plan.down);
            for (std::size_t index = 0; index < hidden.size(); ++index) {
                hidden[index] = quantize(residual[index] + down[index], dtype);
            }
        }

        hidden = rms_norm_block(hidden, batch, *final_norm_);
        auto flat_logits = linear_block(hidden, batch, *lm_head_);
        std::vector<std::vector<float>> output(batch);
        for (std::size_t request = 0; request < batch; ++request) {
            const auto begin = flat_logits.begin() +
                static_cast<std::ptrdiff_t>(request * config.vocab_size);
            output[request].assign(
                begin, begin + static_cast<std::ptrdiff_t>(config.vocab_size)
            );
        }
        return output;
    }

    std::vector<float> forward_token(
        int token_id,
        bool produce_logits,
        std::vector<LayerCache>& caches
    ) {
        const auto& config = descriptor_.config;
        std::vector<float> hidden = embedding(token_id);
        const DType dtype = embedding_weight_->dtype;
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            const auto& plan = layers_[layer];
            auto residual = hidden;
            auto normalized = rms_norm(hidden, *plan.input_norm);
            std::vector<float> query;
            std::vector<float> key;
            std::vector<float> value;
            linear_triple(
                normalized,
                *plan.query,
                *plan.key,
                *plan.value,
                query,
                key,
                value
            );
            auto& cache = caches[layer];
            apply_rope(query, key, cache.length, dtype);
            cache.key.insert(cache.key.end(), key.begin(), key.end());
            cache.value.insert(cache.value.end(), value.begin(), value.end());
            ++cache.length;
            auto attended = attention(query, cache, dtype, cache.length);
            auto projected = linear(attended, *plan.attention_output);
            for (std::size_t index = 0; index < hidden.size(); ++index) {
                hidden[index] = quantize(residual[index] + projected[index], dtype);
            }

            residual = hidden;
            normalized = rms_norm(hidden, *plan.post_attention_norm);
            std::vector<float> gate;
            std::vector<float> up;
            linear_pair(normalized, *plan.gate, *plan.up, gate, up);
            for (std::size_t index = 0; index < gate.size(); ++index) {
                const float silu = quantize(
                    gate[index] / (1.0F + std::exp(-gate[index])), dtype
                );
                gate[index] = quantize(silu * up[index], dtype);
            }
            auto down = linear(gate, *plan.down);
            for (std::size_t index = 0; index < hidden.size(); ++index) {
                hidden[index] = quantize(residual[index] + down[index], dtype);
            }
        }
        if (!produce_logits) return {};
        hidden = rms_norm(hidden, *final_norm_);
        return linear(hidden, *lm_head_);
    }

    Descriptor descriptor_;
    MappedFile weights_;
    Parallel parallel_;
    std::size_t threads_;
    std::size_t head_dim_{};
    std::vector<LayerCache> caches_;
    KernelSet kernels_{portable_kernel_set()};
    std::size_t matrix_passes_{};
    mutable std::size_t tensor_lookups_{};
    const Tensor* embedding_weight_{};
    const Tensor* final_norm_{};
    const Tensor* lm_head_{};
    std::vector<LayerPlan> layers_;
    std::vector<float> rope_cos_;
    std::vector<float> rope_sin_;
    mutable std::vector<float> attention_scores_;
};

std::vector<float> matvec_bf16(
    std::span<const std::uint16_t> weights,
    std::span<const std::uint16_t> values,
    std::size_t rows,
    std::size_t columns
) {
    std::vector<float> output(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0; column < columns; ++column) {
            sum += bf16_to_float(weights[row * columns + column]) *
                   bf16_to_float(values[column]);
        }
        output[row] = round_bf16(sum);
    }
    return output;
}

int self_test() {
    const std::vector<float> weight_values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> input_values{0.5F, 1.5F};
    std::vector<std::uint16_t> weights;
    std::vector<std::uint16_t> input;
    for (float value : weight_values) weights.push_back(float_to_bf16(value));
    for (float value : input_values) input.push_back(float_to_bf16(value));
    const auto output = matvec_bf16(weights, input, 2, 2);
    std::cout << std::setprecision(9)
              << "{\"engine\":\"strpot-native\","
              << "\"dtype\":\"bfloat16\","
              << "\"kernel_family\":\"" << portable_kernel_set().name << "\","
              << "\"matvec\":[" << output[0] << ',' << output[1] << "]}\n";
    return 0;
}

std::vector<int> parse_tokens(const std::string& value) {
    std::vector<int> tokens;
    for (const auto& token : split(value, ',')) tokens.push_back(std::stoi(token));
    return tokens;
}

std::vector<std::vector<int>> parse_batch_tokens(const std::string& value) {
    std::vector<std::vector<int>> prompts;
    for (const auto& prompt : split(value, ';')) prompts.push_back(parse_tokens(prompt));
    return prompts;
}

int generate(int argc, char** argv) {
    if (argc != 7) {
        throw std::runtime_error(
            "usage: strpot-native --generate DESCRIPTOR CHECKPOINT TOKENS MAX_NEW THREADS"
        );
    }
    const auto descriptor = read_descriptor(argv[2]);
    const std::size_t threads = std::stoull(argv[6]);
    LlamaModel model(descriptor, argv[3], threads);
    const auto generation = model.generate(
        parse_tokens(argv[4]), std::stoull(argv[5])
    );
    const std::size_t decoded = generation.tokens.size() > 1
        ? generation.tokens.size() - 1
        : 0;
    const double throughput = generation.decode_seconds > 0.0
        ? static_cast<double>(decoded) / generation.decode_seconds
        : 0.0;
    std::cout << std::setprecision(12)
              << "{\"engine\":\"strpot-native\","
              << "\"weight_dtype\":\"" << model.weight_dtype() << "\","
              << "\"kernel_family\":\"" << model.kernel_family() << "\","
              << "\"threads\":" << model.threads() << ','
              << "\"rope_table_entries\":" << model.rope_table_entries() << ','
              << "\"attention_workspace_floats\":"
              << model.attention_workspace_floats() << ','
              << "\"prefill_seconds\":" << generation.prefill_seconds << ','
              << "\"prefill_matrix_passes\":" << generation.prefill_matrix_passes << ','
              << "\"runtime_tensor_lookups\":" << generation.runtime_tensor_lookups << ','
              << "\"decode_parallel_dispatches\":"
              << generation.decode_parallel_dispatches << ','
              << "\"decode_seconds\":" << generation.decode_seconds << ','
              << "\"decode_tokens_per_second\":" << throughput << ','
              << "\"final_logits_hash\":\"" << generation.final_logits_hash << "\","
              << "\"final_kv_hash\":\"" << generation.final_kv_hash << "\","
              << "\"final_kv_lengths\":[";
    for (std::size_t index = 0; index < generation.final_kv_lengths.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << generation.final_kv_lengths[index];
    }
    std::cout << "],\"frontier_logits_hashes\":[";
    for (std::size_t index = 0; index < generation.frontier_logits_hashes.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '"' << generation.frontier_logits_hashes[index] << '"';
    }
    std::cout << "],\"frontier_kv_hashes\":[";
    for (std::size_t index = 0; index < generation.frontier_kv_hashes.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '"' << generation.frontier_kv_hashes[index] << '"';
    }
    std::cout << "],\"inter_token_seconds\":[";
    for (std::size_t index = 0; index < generation.inter_token_seconds.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << generation.inter_token_seconds[index];
    }
    std::cout << "],"
              << "\"tokens\":[";
    for (std::size_t index = 0; index < generation.tokens.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << generation.tokens[index];
    }
    std::cout << "]}\n";
    return 0;
}

int generate_token_wave_cli(int argc, char** argv) {
    if (argc != 9) {
        throw std::runtime_error(
            "usage: strpot-native --generate-token-wave DESCRIPTOR CHECKPOINT TOKENS MAX_NEW MAX_PROPOSALS THREADS ADVERSARIAL"
        );
    }
    const auto descriptor = read_descriptor(argv[2]);
    LlamaModel model(descriptor, argv[3], std::stoull(argv[7]));
    const auto wave = model.generate_token_wave(
        parse_tokens(argv[4]), std::stoull(argv[5]), std::stoull(argv[6]),
        std::string_view(argv[8]) == "1"
    );
    const auto& generation = wave.generation;
    const double committed_per_traversal = wave.target_weight_traversals > 0
        ? static_cast<double>(wave.committed_decode_tokens) /
              static_cast<double>(wave.target_weight_traversals)
        : 0.0;
    std::cout << std::setprecision(12)
              << "{\"engine\":\"strpot-native-token-wave\","
              << "\"kernel_family\":\"" << model.kernel_family() << "\","
              << "\"weight_dtype\":\"" << model.weight_dtype() << "\","
              << "\"threads\":" << model.threads() << ','
              << "\"prefill_seconds\":" << generation.prefill_seconds << ','
              << "\"decode_seconds\":" << generation.decode_seconds << ','
              << "\"target_weight_traversals\":" << wave.target_weight_traversals << ','
              << "\"committed_decode_tokens\":" << wave.committed_decode_tokens << ','
              << "\"committed_tokens_per_target_weight_traversal\":"
              << committed_per_traversal << ','
              << "\"rolled_back_tokens\":" << wave.rolled_back_tokens << ','
              << "\"speculative_traversals\":" << wave.speculative_traversals << ','
              << "\"fallback_traversals\":" << wave.fallback_traversals << ','
              << "\"transactional_snapshot_bytes_copied\":"
              << wave.transactional_snapshot_bytes_copied << ','
              << "\"rollback_verified\":"
              << (wave.rollback_verified ? "true" : "false") << ','
              << "\"final_logits_hash\":\"" << generation.final_logits_hash << "\","
              << "\"final_kv_hash\":\"" << generation.final_kv_hash << "\","
              << "\"final_kv_lengths\":[";
    for (std::size_t index = 0; index < generation.final_kv_lengths.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << generation.final_kv_lengths[index];
    }
    std::cout << "],\"frontier_logits_hashes\":[";
    for (std::size_t index = 0; index < generation.frontier_logits_hashes.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '"' << generation.frontier_logits_hashes[index] << '"';
    }
    std::cout << "],\"frontier_kv_hashes\":[";
    for (std::size_t index = 0; index < generation.frontier_kv_hashes.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '"' << generation.frontier_kv_hashes[index] << '"';
    }
    std::cout << "],\"acceptance_lengths\":[";
    for (std::size_t index = 0; index < wave.acceptance_lengths.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << wave.acceptance_lengths[index];
    }
    std::cout << "],\"traversal_seconds\":[";
    for (std::size_t index = 0; index < wave.traversal_seconds.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << wave.traversal_seconds[index];
    }
    std::cout << "],\"tokens\":[";
    for (std::size_t index = 0; index < generation.tokens.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << generation.tokens[index];
    }
    std::cout << "]}\n";
    return 0;
}

int generate_batch_cli(int argc, char** argv) {
    if (argc != 7) {
        throw std::runtime_error(
            "usage: strpot-native --generate-batch DESCRIPTOR CHECKPOINT PROMPTS MAX_NEW THREADS"
        );
    }
    const auto descriptor = read_descriptor(argv[2]);
    const std::size_t threads = std::stoull(argv[6]);
    LlamaModel model(descriptor, argv[3], threads);
    const auto generation = model.generate_batch(
        parse_batch_tokens(argv[4]), std::stoull(argv[5])
    );
    std::size_t decoded = 0;
    for (const auto& tokens : generation.tokens) {
        if (tokens.size() > 1) decoded += tokens.size() - 1;
    }
    const double throughput = generation.decode_seconds > 0.0
        ? static_cast<double>(decoded) / generation.decode_seconds
        : 0.0;
    std::cout << std::setprecision(12)
              << "{\"engine\":\"strpot-native\","
              << "\"kernel_family\":\"" << model.kernel_family() << "\","
              << "\"weight_dtype\":\"" << model.weight_dtype() << "\","
              << "\"threads\":" << model.threads() << ','
              << "\"batch_size\":" << generation.tokens.size() << ','
              << "\"activation_panel_width\":" << model.activation_panel_width() << ','
              << "\"prefill_seconds\":" << generation.prefill_seconds << ','
              << "\"decode_seconds\":" << generation.decode_seconds << ','
              << "\"decode_matrix_passes\":" << generation.decode_matrix_passes << ','
              << "\"decode_parallel_dispatches\":"
              << generation.decode_parallel_dispatches << ','
              << "\"aggregate_decode_tokens_per_second\":" << throughput << ','
              << "\"tokens\":[";
    for (std::size_t request = 0; request < generation.tokens.size(); ++request) {
        if (request != 0) std::cout << ',';
        std::cout << '[';
        for (std::size_t index = 0; index < generation.tokens[request].size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << generation.tokens[request][index];
        }
        std::cout << ']';
    }
    std::cout << "]}\n";
    return 0;
}

}  // namespace strpot

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
            return strpot::self_test();
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--generate") {
            return strpot::generate(argc, argv);
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--generate-token-wave") {
            return strpot::generate_token_wave_cli(argc, argv);
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--generate-batch") {
            return strpot::generate_batch_cli(argc, argv);
        }
        std::cerr << "usage: strpot-native --self-test | --generate ... | --generate-token-wave ... | --generate-batch ...\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
