async function loadWasm() {
    const modulePath = `./../pkg/dist-std/purecv_wasm.js`;
    try {
        const module = await import(modulePath);
        console.log(module);

        const init = module.default;

        // Initialize the wasm instance — must await before using exports
        await init();
        module.init_purecv();

        try {
            module.init_panic_hook();
        } catch (e) {
            // Already initialized or not available
        }

        console.log(`[purecv] v${module.get_version()} initialized. Memory isolated.`);
        console.log("WASM module loaded successfully.");
        module.print_version();
    } catch (err) {
        console.error("Failed to load WASM module:", err);
    }
}

async function run() {
    await loadWasm();
    // Additional initialization or test code can go here
}

run();