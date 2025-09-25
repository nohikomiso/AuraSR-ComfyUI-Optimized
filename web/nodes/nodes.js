import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.AuraSR_PurgeCache",
    async nodeCreated(node) {
        if (node.comfyClass === "AuraSR.AuraSRUpscaler") {
            addAuraSRUI(node);
        }
    }
});

function addAuraSRUI(node) {
    // Add a button widget
    node.addWidget("button", "Purge Cache", null, async () => {
        try {
            const resp = await fetch("/aura_sr/purge_cache", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({})
            });
            const data = await resp.json();
            if (data.success) {
                console.log("[AuraSR-ComfyUI] Cache purged successfully.");
                alert("AuraSR cache purged!");
            } else {
                console.error("[AuraSR-ComfyUI] Failed to purge cache:", data.error);
                alert("Failed to purge cache: " + (data.error || "unknown error"));
            }
        } catch (e) {
            console.error("[AuraSR-ComfyUI] purge request error:", e);
            alert("Error purging cache: " + e);
        }
    });

    node.setDirtyCanvas(true);
}
