{
  description = "sam-3d-body flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        rooted = exec:
          builtins.concatStringsSep "\n"
          [
            ''REPO_ROOT="$(git rev-parse --show-toplevel)"''
            exec
          ];

        scripts = {
          dx = {
            exec = rooted ''$EDITOR "$REPO_ROOT"/flake.nix'';
            description = "Edit flake.nix";
          };
        };

        scriptPackages =
          pkgs.lib.mapAttrs
          (
            name: script:
              pkgs.writeShellApplication {
                inherit name;
                text = script.exec;
                runtimeInputs = script.deps or [];
                runtimeEnv = script.env or {};
              }
          )
          scripts;
      in {
        devShells = let
          shellHook = ''
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            export NIX_SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
          '';

          env = {
            RUST_BACKTRACE = "1";
            DEV = "1";
            LOCAL = "1";
            PYTHONNOUSERSITE = "1";
            CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
            LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.cudaPackages.cudatoolkit
              pkgs.stdenv.cc.cc
              pkgs.glibc
              pkgs.glib
              pkgs.libGL
              pkgs.openssl
              pkgs.cacert
              pkgs.libglvnd
              pkgs.xorg.libX11
              pkgs.xorg.libXrandr
              pkgs.xorg.libXinerama
              pkgs.xorg.libXcursor
              pkgs.xorg.libXi
              pkgs.opencv4
            ]}";
          };

          corePackages = with pkgs; [
            # Nix development tools
            alejandra
            nixd
            nil
            statix
            deadnix
          ];

          systemPackages = with pkgs; [
            pkg-config
            protobuf
            openssl
            opencv4
            cacert
            ninja
            stdenv.cc
            libglvnd
          ];

          # Platform-specific packages
          linuxPackages = [
            pkgs.libGL
            pkgs.mesa
            pkgs.xorg.libX11
            pkgs.xorg.libXrandr
            pkgs.xorg.libXinerama
            pkgs.xorg.libXcursor
            pkgs.xorg.libXi
          ];

          shell-packages =
            corePackages
            ++ systemPackages
            ++ linuxPackages
            ++ builtins.attrValues scriptPackages;
        in {
          default = pkgs.mkShell {
            inherit shellHook env;
            packages = shell-packages;
          };
        };
      }
    );
}
