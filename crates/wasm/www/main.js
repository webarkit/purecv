async function loadWasm() {
     const modulePath = `./../pkg/dist-std/purecv_wasm.js`;
    try {
        const module = await import(modulePath);
        const init = module.default;
        
        // Initialize the wasm instance
        init();
        module.init_wasm();

        try {
            module.init_panic_hook();
        } catch (e) {
            // Already initialized or not available
        }
        
        console.log(`[purecv] ${version} initialized. Memory isolated.`);
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