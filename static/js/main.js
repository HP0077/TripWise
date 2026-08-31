// ==============================================================================
// TripWise - Frontend Interaction & Recommendation Script
// Handles multi-step form transitions, API calls, and Wikipedia image loading.
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Sections
    const heroSection = document.getElementById('hero-section');
    const formSection = document.getElementById('form-section');
    const resultsSection = document.getElementById('results-section');
    
    // DOM Elements - Action Buttons
    const startPlanningBtn = document.getElementById('start-planning-btn');
    const backToLandingBtn = document.getElementById('back-to-landing-btn');
    const modifyPreferencesBtn = document.getElementById('modify-preferences-btn');
    const submitBtn = document.getElementById('submit-btn');
    const btnSpinner = submitBtn.querySelector('.btn-spinner');
    const btnText = submitBtn.querySelector('span');
    
    // DOM Elements - Form & Results
    const travelForm = document.getElementById('travel-form');
    const destContainer = document.getElementById('destinations-container');
    const summaryElement = document.getElementById('results-summary');

    // Navigation: 1. Landing Page -> Preferences Form
    startPlanningBtn.addEventListener('click', () => {
        heroSection.classList.add('hidden');
        formSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Navigation: 2. Preferences Form -> Back to Landing
    backToLandingBtn.addEventListener('click', () => {
        formSection.classList.add('hidden');
        heroSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Navigation: 3. Results -> Modify Preferences
    modifyPreferencesBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        formSection.classList.remove('hidden');
        heroSection.classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Form Submission Handler
    travelForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        // Display loading state on submit button
        btnSpinner.classList.remove('hidden');
        btnText.textContent = "Matching Destinations...";
        submitBtn.disabled = true;

        // Collect all form values
        const formData = new FormData(travelForm);
        const requestPayload = {
            budget: formData.get('budget'),
            season: formData.get('season'),
            activity: formData.get('activity'),
            safety: formData.get('safety'),
            experiences: formData.getAll('experiences'),
            scenery: formData.getAll('scenery'),
            region: formData.getAll('region')
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload)
            });

            const data = await response.json();

            if (data.success && data.destinations && data.destinations.length > 0) {
                await renderDestinationResults(data.destinations, data.summary);
            } else {
                alert('No matching destinations found for those specific criteria. Try broadening your selections!');
            }
        } catch (error) {
            console.error('Error fetching model predictions:', error);
            alert('Failed to connect to the prediction server. Please make sure the Flask backend is running.');
        } finally {
            // Reset button loading state
            btnSpinner.classList.add('hidden');
            btnText.textContent = "Find Matching Destinations";
            submitBtn.disabled = false;
        }
    });

    // Renders the predicted destination cards with Wikipedia images
    async function renderDestinationResults(destinations, summaryText) {
        destContainer.innerHTML = '';
        
        if (summaryElement && summaryText) {
            summaryElement.textContent = summaryText;
        }

        // Show results section
        formSection.classList.add('hidden');
        heroSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });

        // Reliable fallback image if Wikipedia has no thumbnail
        const defaultFallbackImage = 'https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=600&q=80';

        // Loop through recommendations and fetch real photo for each
        for (const item of destinations) {
            const nameParts = item.name.split(',');
            const cityName = nameParts[0].trim();
            const locationSubtitle = nameParts.slice(1).join(',').trim() || cityName;

            let photoUrl = defaultFallbackImage;

            try {
                // Fetch high-res image from Wikipedia Search Generator API
                const wikiEndpoint = `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(cityName)}&gsrlimit=1&prop=pageimages&format=json&pithumbsize=600&origin=*`;
                const wikiResponse = await fetch(wikiEndpoint);
                const wikiResult = await wikiResponse.json();

                if (wikiResult.query && wikiResult.query.pages) {
                    const pages = wikiResult.query.pages;
                    const pageKey = Object.keys(pages)[0];
                    if (pages[pageKey].thumbnail && pages[pageKey].thumbnail.source) {
                        photoUrl = pages[pageKey].thumbnail.source;
                    }
                }
            } catch (err) {
                console.warn(`Wikipedia image not found for ${cityName}, using default fallback.`);
            }

            // Create and append the destination card
            const destinationCard = document.createElement('div');
            destinationCard.className = 'dest-card';
            destinationCard.innerHTML = `
                <div class="dest-img-wrap">
                    <img src="${photoUrl}" alt="${item.name}" class="dest-img" loading="lazy">
                    <span class="dest-match-tag">🔥 ${item.match}% Match</span>
                </div>
                <div class="dest-info">
                    <div>
                        <h3 class="dest-name">${cityName}</h3>
                        <p class="dest-country">📍 ${locationSubtitle}</p>
                    </div>
                </div>
            `;
            destContainer.appendChild(destinationCard);
        }
    }
});
