document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("img").forEach(function (img) {
        const src = img.getAttribute("src") || "";
        if (
            src.includes("logo.svg") ||
            img.classList.contains("hero-gallery-img")
        ) {
            // Don't lazy-load logo or hero image
            return;
        }
        img.setAttribute("loading", "lazy");
    });
    document.body.classList.add("ready");
});