use std::{path::PathBuf, process::Command};

fn main() {
    // 1. Use `swift-bridge-build` to generate Swift/C FFI glue.
    //    You can also use the `swift-bridge` CLI.
    let bridge_files = vec!["src/swift.rs"];
    swift_bridge_build::parse_bridges(bridge_files)
        .write_all_concatenated(swift_bridge_out_dir(), "rust-calls-swift");
    export_bridge_functions();

    // 2. Compile Swift library
    compile_swift();

    // 3. Link to Swift library
    println!("cargo:rustc-link-lib=static=swift-library");
    println!(
        "cargo:rustc-link-search={}",
        swift_library_static_lib_dir().to_str().unwrap()
    );

    // Without this we will get warnings about not being able to find dynamic libraries, and then
    // we won't be able to compile since the Swift static libraries depend on them:
    // For example:
    // ld: warning: Could not find or use auto-linked library 'swiftCompatibility51'
    // ld: warning: Could not find or use auto-linked library 'swiftCompatibility50'
    // ld: warning: Could not find or use auto-linked library 'swiftCompatibilityDynamicReplacements'
    // ld: warning: Could not find or use auto-linked library 'swiftCompatibilityConcurrency'
    let xcode_path = if let Ok(output) = std::process::Command::new("xcode-select")
        .arg("--print-path")
        .output()
    {
        String::from_utf8(output.stdout.as_slice().into())
            .unwrap()
            .trim()
            .to_string()
    } else {
        "/Applications/Xcode.app/Contents/Developer".to_string()
    };
    println!(
        "cargo:rustc-link-search={}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx/",
        &xcode_path
    );
    println!("cargo:rustc-link-search={}", "/usr/lib/swift");
}

/// Make the generated `@_cdecl` glue `public`.
///
/// swift-bridge emits it as internal `func`, which swiftc -O gives hidden visibility.
/// swiftbuild (SwiftPM 6.4's default) merges the objects with `ld -r` before archiving,
/// and that turns hidden symbols into locals, so the final link fails with undefined
/// `___swift_bridge__$...`. Public keeps them default visibility through the merge; the
/// native build system archives the objects as they are, so it is unaffected either way.
fn export_bridge_functions() {
    let path = generated_code_dir()
        .join("rust-calls-swift")
        .join("rust-calls-swift.swift");
    let source = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", path.display(), e);
        std::process::exit(1);
    });

    let mut out = String::with_capacity(source.len());
    let mut after_cdecl = false;
    for line in source.split_inclusive('\n') {
        if after_cdecl && line.starts_with("func ") {
            out.push_str("public ");
        }
        after_cdecl = line.starts_with("@_cdecl(");
        out.push_str(line);
    }

    std::fs::write(&path, out).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {}", path.display(), e);
        std::process::exit(1);
    });
}

fn compile_swift() {
    let swift_package_dir = manifest_dir().join("swift-library");

    let triple = std::env::var("TARGET").unwrap();
    let parts = triple.split("-").collect::<Vec<_>>();
    // SwiftPM takes Apple's arch names. The native build system also accepted Rust's
    // `aarch64`, but swiftbuild (the default from SwiftPM 6.4) skips an arch it does not
    // know and reports a successful build with nothing built.
    let arch = match *parts.first().unwrap() {
        "aarch64" => "arm64",
        arch => arch,
    };

    let mut cmd = Command::new("swift");

    cmd.current_dir(swift_package_dir)
        .arg("build")
        .args(&["--arch", &arch])
        .args(&["-Xswiftc", "-static"])
        .args(&[
            "-Xswiftc",
            "-import-objc-header",
            "-Xswiftc",
            swift_source_dir()
                .join("bridging-header.h")
                .to_str()
                .unwrap(),
        ]);

    if is_release_build() {
        cmd.args(&["-c", "release"]);
    }

    let child = cmd.spawn().unwrap_or_else(|e| {
        eprintln!("Failed to spawn swift build command: {}", e);
        std::process::exit(1);
    });
    let exit_status = child.wait_with_output().unwrap_or_else(|e| {
        eprintln!("Failed to wait for swift build: {}", e);
        std::process::exit(1);
    });

    if !exit_status.status.success() {
        eprintln!(
            "Swift build failed:\nStderr: {}\nStdout: {}",
            String::from_utf8_lossy(&exit_status.stderr),
            String::from_utf8_lossy(&exit_status.stdout),
        );
        std::process::exit(1);
    }
}

fn swift_bridge_out_dir() -> PathBuf {
    generated_code_dir()
}

fn manifest_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest_dir)
}

fn is_release_build() -> bool {
    std::env::var("PROFILE").unwrap() == "release"
}

fn swift_source_dir() -> PathBuf {
    manifest_dir().join("swift-library/Sources/swift-library")
}

fn generated_code_dir() -> PathBuf {
    swift_source_dir().join("generated")
}

fn swift_library_static_lib_dir() -> PathBuf {
    let debug_or_release = if is_release_build() {
        "release"
    } else {
        "debug"
    };

    manifest_dir().join(format!("swift-library/.build/{}", debug_or_release))
}
