# build stage
FROM registry.fedoraproject.org/fedora:44 AS build

WORKDIR /app

# ROCm 7.14 repository (TheRock release stream)
RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[rocm]
name=ROCm 7.14.0
baseurl=https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg
REPO
EOF

RUN dnf -y --nodocs --setopt=install_weak_deps=False \
  install \
  make gcc gcc-c++ cmake git ninja-build rdma-core-devel \
  amdrocm-core-devel7.14-gfx1151 \
  && dnf clean all && rm -rf /var/cache/dnf/*

ENV ROCM_PATH=/opt/rocm \
  HIP_PATH=/opt/rocm \
  PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH \
  LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib

COPY .. .
RUN make base.en CMAKE_ARGS="-DGGML_HIP=1 -DAMDGPU_TARGETS=\"gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1200;gfx1201\""

# runtime stage
FROM registry.fedoraproject.org/fedora-minimal:44 AS runtime

WORKDIR /app

# ROCm 7.14 repository (TheRock release stream)
RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[rocm]
name=ROCm 7.14.0
baseurl=https://repo.amd.com/rocm/packages-multi-arch/rhel10/x86_64
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg
REPO
EOF

RUN microdnf -y --nodocs --setopt=install_weak_deps=0 \
  install \
  bash ca-certificates libatomic libstdc++ libgcc libgomp libibverbs \
  amdrocm-runtime7.14 \
  amdrocm-blas7.14-gfx1100 \
  amdrocm-blas7.14-gfx1101 \
  amdrocm-blas7.14-gfx1102 \
  amdrocm-blas7.14-gfx1103 \
  amdrocm-blas7.14-gfx1150 \
  amdrocm-blas7.14-gfx1151 \
  amdrocm-blas7.14-gfx1152 \
  amdrocm-blas7.14-gfx1200 \
  amdrocm-blas7.14-gfx1201 \
  && microdnf clean all && rm -rf /var/cache/dnf/* \
  && ln -s core-7.14 /opt/rocm/core \
  && ln -s core-7.14/bin /opt/rocm/bin \
  && ln -s core-7.14/include /opt/rocm/include \
  && ln -s core-7.14/lib /opt/rocm/lib \
  && ln -s core-7.14/libexec /opt/rocm/libexec \
  && ln -s core-7.14/lib/llvm /opt/rocm/llvm \
  && ln -s core-7.14/share /opt/rocm/share \
  && ln -s core-7.14/lib/llvm/amdgcn /opt/rocm/amdgcn

ENV ROCM_PATH=/opt/rocm \
  HIP_PATH=/opt/rocm \
  PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH \
  LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib

COPY --from=build /app /app
ENV PATH=/app/build/bin:$PATH
ENTRYPOINT [ "bash", "-c" ]
