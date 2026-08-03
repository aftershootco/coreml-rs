{
  description = "coreml-rs — CoreML bindings for Rust via swift-bridge";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    swiftix.url = "github:stillwind-ai/swiftix";
  };

  outputs = { nixpkgs, flake-utils, swiftix, ... }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" "x86_64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            swiftix.packages.${system}.swift-6_3
            pkgs.apple-sdk_15
            pkgs.cargo
            pkgs.rustc
            pkgs.clippy
            pkgs.rustfmt
            pkgs.rust-analyzer
            pkgs.libiconv
            pkgs.pkg-config
          ];

          env.RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

          shellHook = ''
            export SDKROOT="$(xcrun --show-sdk-path 2>/dev/null || echo ${pkgs.apple-sdk_15.sdkroot})"
          '';
        };
      });
}
