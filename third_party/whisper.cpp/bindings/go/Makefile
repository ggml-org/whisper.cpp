ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

GGML_METAL_PATH_RESOURCES := $(abspath ../..)
BUILD_DIR := build_go
MODELS_DIR := models
EXAMPLES_DIR := $(wildcard examples/*)
INCLUDE_PATH := $(abspath ../../include):$(abspath ../../ggml/include)
LIBRARY_PATH := $(abspath ../../${BUILD_DIR}/src):$(abspath ../../${BUILD_DIR}/ggml/src)

ifeq ($(GGML_CUDA),1)
	LIBRARY_PATH := $(LIBRARY_PATH):$(CUDA_PATH)/targets/$(UNAME_M)-linux/lib/
	BUILD_FLAGS := -ldflags "-extldflags '-lcudart -lcuda -lcublas'"
endif

ifeq ($(UNAME_S),Darwin)
	LIBRARY_PATH := $(LIBRARY_PATH):$(abspath ../../${BUILD_DIR}/ggml/src/ggml-blas):$(abspath ../../${BUILD_DIR}/ggml/src/ggml-metal)
	EXT_LDFLAGS := -framework Foundation -framework Metal -framework MetalKit -lggml-metal -lggml-blas
endif

all: clean whisper examples

whisper: mkdir
	cmake -S ../.. -B ../../${BUILD_DIR} \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF
	cmake --build ../../${BUILD_DIR} --target whisper

test: model-small whisper modtidy
ifeq ($(UNAME_S),Darwin)
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} GGML_METAL_PATH_RESOURCES=${GGML_METAL_PATH_RESOURCES} go test -ldflags "-extldflags '$(EXT_LDFLAGS)'" -v .
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} GGML_METAL_PATH_RESOURCES=${GGML_METAL_PATH_RESOURCES} go test -ldflags "-extldflags '$(EXT_LDFLAGS)'" -v ./pkg/whisper/...
else
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go test -v .
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go test -v ./pkg/whisper/...
endif

examples: $(EXAMPLES_DIR)

model-small: mkdir examples/go-model-download
	@${BUILD_DIR}/go-model-download -out models ggml-small.en.bin

$(EXAMPLES_DIR): mkdir whisper modtidy
	@echo Build example $(notdir $@)
ifeq ($(UNAME_S),Darwin)
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} GGML_METAL_PATH_RESOURCES=${GGML_METAL_PATH_RESOURCES} go build ${BUILD_FLAGS} -ldflags "-extldflags '$(EXT_LDFLAGS)'" -o ${BUILD_DIR}/$(notdir $@) ./$@
else
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go build ${BUILD_FLAGS} -o ${BUILD_DIR}/$(notdir $@) ./$@
endif

mkdir:
	@echo Mkdir ${BUILD_DIR}
	@install -d ${BUILD_DIR}
	@echo Mkdir ${MODELS_DIR}
	@install -d ${MODELS_DIR}

modtidy:
	@go mod tidy

clean:
	@echo Clean
	@rm -fr $(BUILD_DIR)
	@go clean
