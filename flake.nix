{
  description = "coreml-rs — CoreML bindings for Rust via swift-bridge";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    nix-github-actions = {
      url = "github:nix-community/nix-github-actions";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
    swiftix.url = "github:stillwind-ai/swiftix";
  };

  outputs = {
    self,
    crane,
    flake-utils,
    nixpkgs,
    rust-overlay,
    advisory-db,
    nix-github-actions,
    swiftix,
    ...
  }:
    flake-utils.lib.eachSystem ["aarch64-darwin" "x86_64-darwin"] (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
          ];
        };
        inherit (pkgs) lib;
        cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
        name = cargoToml.package.name;

        swift = swiftix.packages.${system}.swift-6_3;

        stableToolchain = pkgs.rust-bin.stable.latest.default;
        stableToolchainWithRustAnalyzer = stableToolchain.override {
          extensions = ["rust-src" "rust-analyzer"];
        };
        craneLib = (crane.mkLib pkgs).overrideToolchain stableToolchain;

        src = let
          filterBySuffix = path: exts: lib.any (ext: lib.hasSuffix ext path) exts;
          sourceFilters = path: type:
            (craneLib.filterCargoSources path type)
            || filterBySuffix path [".swift" ".h" ".c" ".toml" "Package.resolved" "/LICENSE"];
        in
          lib.cleanSourceWith {
            filter = sourceFilters;
            src = ./.;
          };

        commonArgs = {
          inherit src;
          pname = name;
          doCheck = false;
          nativeBuildInputs = [swift];
          buildInputs = with pkgs; [
            libiconv
            apple-sdk_15
          ];
          SWIFTPM_DISABLE_SANDBOX = 1;
          # SwiftPM needs a writable HOME for its module/manifest caches.
          preConfigure = ''
            export HOME=$(mktemp -d)
          '';
        };
        cargoArtifacts = craneLib.buildPackage commonArgs;
      in {
        checks = {
          "${name}-clippy" = craneLib.cargoClippy (commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            });
          "${name}-docs" = craneLib.cargoDoc (commonArgs // {inherit cargoArtifacts;});
          "${name}-fmt" = craneLib.cargoFmt {inherit src;};
          "${name}-toml-fmt" = craneLib.taploFmt {
            src = pkgs.lib.sources.sourceFilesBySuffices src [".toml"];
          };
          # Audit dependencies
          "${name}-audit" = craneLib.cargoAudit {
            inherit src advisory-db;
          };

          # Audit licenses
          "${name}-deny" = craneLib.cargoDeny {
            inherit src;
          };
          "${name}-nextest" = craneLib.cargoNextest (commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
            });
        };

        packages = let
          pkg = craneLib.buildPackage (commonArgs // {inherit cargoArtifacts;});
        in {
          "${name}" = pkg;
          default = pkg;
        };

        devShells.default = pkgs.mkShell (commonArgs
          // {
            packages = [
              stableToolchainWithRustAnalyzer
              swift
              pkgs.cargo-nextest
              pkgs.cargo-deny
            ];
            RUST_SRC_PATH = "${stableToolchainWithRustAnalyzer}/lib/rustlib/src/rust/library";
          });
      }
    )
    // {
      githubActions = nix-github-actions.lib.mkGithubMatrix {
        checks = nixpkgs.lib.getAttrs ["aarch64-darwin"] self.checks;
      };
    };
}
