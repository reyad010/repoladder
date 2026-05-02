# Builds the shared-memory bridge (shm_bridge.so) and, when SGX=1, the Gramine
# pytorch.manifest.sgx for running tee_server.py inside an SGX enclave.
#
#   make             # compiles shm_bridge.so only (gpu-cpu mode)
#   make sgx         # also builds pytorch.manifest.sgx (gpu-tee mode)
#   make clean

SHELL := /bin/bash
THIS_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

CXX      ?= g++
CXXFLAGS ?= -O3 -fPIC -shared

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)
GRAMINE_LOG_LEVEL ?= error

PYTHON_BIN    ?= $(shell which python3)
VENV_DIR      ?= $(abspath $(THIS_DIR)..)
ARTIFACT_ROOT ?= $(abspath $(THIS_DIR)..)
# X.Y of $(PYTHON_BIN) (e.g. 3.10, 3.12) — used to fill in venv lib paths.
PYTHON_VER    ?= $(shell $(PYTHON_BIN) -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')

.PHONY: all
all: shm_bridge.so

shm_bridge.so: shm_bridge.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# ---- Gramine SGX ---------------------------------------------------------
.PHONY: sgx
sgx: shm_bridge.so pytorch.manifest.sgx pytorch.sig

pytorch.manifest: gramine/pytorch.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(abspath $(PYTHON_BIN)) \
		-Dvenv_dir=$(VENV_DIR) \
		-Dartifact_root=$(ARTIFACT_ROOT) \
		-Dpython_ver=$(PYTHON_VER) \
		$< > $@

pytorch.manifest.sgx pytorch.sig: pytorch.manifest
	gramine-sgx-sign --manifest $< --output pytorch.manifest.sgx

.PHONY: clean
clean:
	$(RM) shm_bridge.so *.token *.sig *.manifest *.manifest.sgx
