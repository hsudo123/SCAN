{
	description = "Python Development Environment";

	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
	};

	outputs = { self, nixpkgs }:
	let
		system = "aarch64-darwin";
		pkgs = import nixpkgs { inherit system; };
	in
	{
		devShells.${system}.default = pkgs.mkShell {
			packages = with pkgs; [
				(python3.withPackages (python-pkgs: with python-pkgs; [
					torch
					torchvision
					timm
					numpy
					opencv4
					albumentationsx
					scipy
					scikit-learn
					scikit-image
					pandas
					matplotlib
					tqdm

				]))
			];

			# shellHook 會在進入 devshell 時自動執行
			shellHook = ''
				echo "============================================"
				echo "  DevShell loaded successfully!       "
				echo "  Python version: $(python --version | head -n1)  "
				echo "============================================"
			'';
		};
	};
}
