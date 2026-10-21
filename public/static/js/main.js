// ==============================================================================
// TripWise - Frontend Script
// Validates the questionnaire, calls /predict, and renders the shortlist with
// photos from Wikipedia.
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('travel-form');
    const submitBtn = document.getElementById('submit-btn');
    const list = document.getElementById('destinations');
    const summary = document.getElementById('shortlist-summary');
    const note = document.getElementById('shortlist-note');
    const shortlist = document.getElementById('shortlist');
    const emptyMessage = summary.textContent;

    // Clear the "missing answer" state as soon as a question is answered
    form.addEventListener('change', (event) => {
        event.target.closest('.question')?.classList.remove('is-missing');
    });

    form.addEventListener('reset', () => {
        form.querySelectorAll('.is-missing').forEach((q) => q.classList.remove('is-missing'));
        list.innerHTML = '';
        summary.textContent = emptyMessage;
        note.hidden = true;
    });

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        // Single-choice questions must be answered
        const missing = [...form.querySelectorAll('.question[data-required]')]
            .filter((q) => !q.querySelector('input:checked'));
        missing.forEach((q) => q.classList.add('is-missing'));
        if (missing.length) {
            missing[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
            missing[0].querySelector('input').focus({ preventScroll: true });
            return;
        }

        const formData = new FormData(form);
        const payload = {
            budget: formData.get('budget'),
            season: formData.get('season'),
            activity: formData.get('activity'),
            safety: formData.get('safety'),
            popularity: formData.get('popularity'),
            experiences: formData.getAll('experiences'),
            scenery: formData.getAll('scenery'),
            region: formData.getAll('region')
        };

        submitBtn.disabled = true;
        submitBtn.textContent = 'Finding matches…';
        list.classList.add('is-loading');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || `Server returned ${response.status}`);
            }
            await renderShortlist(data.destinations, data.summary);
        } catch (error) {
            console.error('Prediction failed:', error);
            list.innerHTML = '';
            note.hidden = true;
            summary.textContent = 'Something went wrong getting your matches. Check that the server is running and try again.';
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Show my shortlist';
            list.classList.remove('is-loading');
        }
    });

    async function renderShortlist(destinations, summaryText) {
        // Fetch all photos in parallel so the list appears at once
        const photos = await Promise.all(destinations.map((d) => findWikipediaPage(d.name)));

        list.innerHTML = '';
        summary.textContent = summaryText;
        note.hidden = false;

        destinations.forEach((item, index) => {
            const [city, ...rest] = item.name.split(',').map((part) => part.trim());
            const place = rest.join(', ');
            const page = photos[index];

            const ticket = document.createElement('li');
            ticket.className = 'ticket';
            ticket.style.animationDelay = `${index * 50}ms`;
            ticket.innerHTML = `
                <div class="ticket-photo">
                    <span class="ticket-rank"></span>
                </div>
                <div class="ticket-body">
                    <h3 class="ticket-name"><a target="_blank" rel="noopener"></a></h3>
                    <p class="ticket-place"></p>
                    <div class="fit">
                        <div class="fit-bar"><span></span></div>
                        <span class="fit-value"></span>
                    </div>
                </div>
            `;

            ticket.querySelector('.ticket-rank').textContent = index + 1;
            const link = ticket.querySelector('.ticket-name a');
            link.textContent = city;
            link.href = page.url;
            ticket.querySelector('.ticket-place').textContent = place;
            ticket.querySelector('.fit-bar span').style.width = `${item.match}%`;
            ticket.querySelector('.fit-value').textContent = Math.round(item.match);

            if (page.image) {
                const img = document.createElement('img');
                img.src = page.image;
                img.alt = `Photo of ${city}`;
                img.width = 88;
                img.height = 88;
                img.loading = 'lazy';
                ticket.querySelector('.ticket-photo').prepend(img);
            }

            list.appendChild(ticket);
        });

        if (window.matchMedia('(max-width: 900px)').matches) {
            shortlist.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    // Looks up the best-matching Wikipedia article and its thumbnail for a destination
    async function findWikipediaPage(fullName) {
        const query = fullName.replace(/,/g, ' ');
        const fallback = { url: `https://en.wikipedia.org/w/index.php?search=${encodeURIComponent(query)}`, image: null };
        try {
            const endpoint = 'https://en.wikipedia.org/w/api.php?action=query&format=json&origin=*'
                + '&generator=search&gsrlimit=1&prop=pageimages|info&inprop=url&pithumbsize=200'
                + `&gsrsearch=${encodeURIComponent(query)}`;
            const result = await (await fetch(endpoint)).json();
            const page = Object.values(result.query?.pages || {})[0];
            if (!page) return fallback;
            return { url: page.fullurl || fallback.url, image: page.thumbnail?.source || null };
        } catch (err) {
            return fallback;
        }
    }
});
