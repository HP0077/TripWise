// ==============================================================================
// TripWise - Theme switch
// Dark is the default. Loaded in <head> so a saved light choice applies before
// the first paint; the masthead button toggles and remembers the choice.
// ==============================================================================

(function () {
    const root = document.documentElement;
    const meta = () => document.querySelector('meta[name="theme-color"]');

    function apply(theme) {
        if (theme === 'light') root.dataset.theme = 'light';
        else delete root.dataset.theme;
        meta()?.setAttribute('content', theme === 'light' ? '#f3f0e8' : '#0e1a26');
    }

    try { apply(localStorage.getItem('tripwise-theme')); } catch (err) { /* storage blocked: stay dark */ }

    document.addEventListener('DOMContentLoaded', () => {
        const button = document.getElementById('theme-toggle');
        if (!button) return;
        const label = () => {
            const light = root.dataset.theme === 'light';
            button.textContent = light ? 'Dark' : 'Light';
            button.setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
        };
        label();
        button.addEventListener('click', () => {
            const next = root.dataset.theme === 'light' ? 'dark' : 'light';
            apply(next);
            label();
            try { localStorage.setItem('tripwise-theme', next); } catch (err) { /* not saved */ }
        });
    });
})();
