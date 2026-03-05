{
  description = "pixi env";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { flake-utils, nixpkgs, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        fhs = pkgs.buildFHSEnv {
          name = "pixi-env";
          targetPkgs = _: [ pkgs.pixi ];
          # runScript = "pixi";  # Default command
        };
      in
      {
        devShells.default = fhs;  # Use fhs directly, not fhs.env
      }
    );
}
