{
  description = "inkling-zml — run Thinking Machines' Inkling-2-Mini on the ZML inference stack (Zig + Bazel)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      # allowUnfree: CUDA packages for the `cuda` dev shell.
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f (import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          })
        );
      # ZML master commit pinned in MODULE.bazel (git_override). Keep in sync.
      zmlCommit = "1470ac33f58ad5795076695a6965abf0f38d4272";
      bazelVersion = "9.3.0";
      basePackages = pkgs: [
        # `bazel` wrapper around bazelisk (same as zml's own devenv)
        (pkgs.writeShellScriptBin "bazel" ''exec ${pkgs.bazelisk}/bin/bazelisk "$@"'')
        pkgs.bazelisk
        pkgs.zig # editor/ZLS only; the hermetic Zig 0.17.0 toolchain is fetched by rules_zig
        pkgs.python3
        pkgs.uv
        pkgs.git
        pkgs.gh
        pkgs.curl
        pkgs.jq
        pkgs.cacert
      ];
    in
    {
      packages = forAllSystems (pkgs: {
        # Bazel's process wrapper execs /bin/bash, which does not exist on NixOS.
        # `nix run .#bazel-fhs -- <args>` runs bazelisk inside a bubblewrap
        # namespace where /bin/bash exists; the host fs is mounted read-only,
        # nothing on the host is modified.
        bazel-fhs = pkgs.writeShellScriptBin "bazel" ''
          exec ${pkgs.bubblewrap}/bin/bwrap \
            --bind / / \
            --dev-bind /dev /dev \
            --proc /proc \
            --tmpfs /tmp \
            --bind ${pkgs.bash}/bin/bash /bin/bash \
            --setenv USE_BAZEL_VERSION "${bazelVersion}" \
            -- ${pkgs.bazelisk}/bin/bazelisk "$@"
        '';
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = basePackages pkgs;
          USE_BAZEL_VERSION = bazelVersion;
          shellHook = ''
            echo "inkling-zml dev shell"
            echo "  bazel: $(bazelisk version 2>/dev/null | head -1)"
            echo "  zig:   $(zig version)"
            echo "  zml:   ${zmlCommit}"
            echo "build: bazel build --config=cpu //...   (GPU: --config=cuda, shell .#cuda)"
          '';
        };
        # NVIDIA GPU shell: CUDA toolkit for `bazel build --config=cuda`.
        cuda = pkgs.mkShell {
          packages = (basePackages pkgs) ++ [
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.nccl
          ];
          USE_BAZEL_VERSION = bazelVersion;
          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.nccl
            pkgs.stdenv.cc.cc.lib
          ];
        };
      });
    };
}
