{
  description = "dbt-osmosis dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
    nix2container.url = "github:nlewo/nix2container";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";
    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
  };

  nixConfig = {
    extra-trusted-public-keys =
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.devenv.flakeModule ];
      systems = [
        "x86_64-linux"
        "i686-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      perSystem = { config, self', inputs', pkgs, system, ... }: {
        devenv.shells.default = {
          name = "dbt-osmosis";
          imports = [ ];

          # Base environment
          dotenv.disableHint = true;

          # Base packages
          packages = [
            pkgs.python310
            pkgs.black
            pkgs.isort
            pkgs.jq
            pkgs.yq
            pkgs.ruff
            pkgs.nixfmt
            pkgs.poetry
          ];

          # Utilities
          scripts.fmt.exec = ''
            echo "Formatting..."
            ${pkgs.black}/bin/black src
            ${pkgs.isort}/bin/isort src
            ${pkgs.nixfmt}/bin/nixfmt flake.nix
          '';
          scripts.lint.exec = ''
            echo "Linting..."
            ${pkgs.ruff}/bin/ruff --fix src/**/*.py
          '';

          # Activate venv on shell enter
          enterShell = ''
            PROJECT_ROOT=$(git rev-parse --show-toplevel)
            echo Setting up Python virtual environment...
            [ -d $PROJECT_ROOT/.venv ] || python -m venv $PROJECT_ROOT/.venv
            export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
            ${pkgs.poetry}/bin/poetry install
            eval "$(${pkgs.poetry}/bin/poetry env info --path)/bin/activate"
          '';

          # Languages
          languages.nix.enable = true;
          languages.python.enable = true;

        };
      };
      flake = { };
    };
}
