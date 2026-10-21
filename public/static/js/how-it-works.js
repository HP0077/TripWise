// ==============================================================================
// TripWise - "How it works" page
// Scroll reveals, table-of-contents tracking, the pipeline steps, the feature
// encoder demo and the two evaluation charts.
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const metrics = JSON.parse(document.getElementById('metrics-data').textContent);
    const tooltip = document.getElementById('tooltip');

    setUpReveals();
    setUpTableOfContents();
    setUpPipeline();
    setUpEncoder();
    setUpResultsChart(metrics);
    setUpImportanceChart();

    // ---------- Reveal each chapter once, when it scrolls into view ----------
    function setUpReveals() {
        const chapters = document.querySelectorAll('[data-reveal]');
        if (!('IntersectionObserver' in window)) {
            chapters.forEach((el) => el.classList.add('is-visible'));
            document.querySelectorAll('.chart').forEach((el) => el.classList.add('is-drawn'));
            return;
        }
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add('is-visible');
                // Bars grow after the chapter has faded in
                entry.target.querySelectorAll('.chart').forEach((chart) => {
                    setTimeout(() => chart.classList.add('is-drawn'), 250);
                });
                observer.unobserve(entry.target);
            });
        }, { threshold: 0.12 });
        chapters.forEach((el) => observer.observe(el));
    }

    // ---------- Highlight the chapter currently being read ----------
    function setUpTableOfContents() {
        const links = new Map([...document.querySelectorAll('.toc a')].map((a) => [a.hash.slice(1), a]));
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                links.forEach((link) => link.classList.remove('is-active'));
                links.get(entry.target.id)?.classList.add('is-active');
            });
        }, { rootMargin: '-35% 0px -60% 0px' });
        document.querySelectorAll('.chapter').forEach((chapter) => observer.observe(chapter));

        // Above the first chapter, nothing is being read yet
        const first = document.querySelector('.chapter');
        window.addEventListener('scroll', () => {
            if (first.getBoundingClientRect().top > window.innerHeight * 0.4) {
                links.forEach((link) => link.classList.remove('is-active'));
            }
        }, { passive: true });
    }

    // ---------- Pipeline: six steps as tabs ----------
    function setUpPipeline() {
        const stations = [...document.querySelectorAll('.station')];
        const panel = document.getElementById('station-panel');
        const templates = panel.querySelectorAll('template');
        const content = document.createElement('div');
        panel.appendChild(content);

        function select(index, focus) {
            stations.forEach((station, i) => {
                const selected = i === index;
                station.setAttribute('aria-selected', selected);
                station.tabIndex = selected ? 0 : -1;
            });
            panel.setAttribute('aria-labelledby', stations[index].id);
            content.replaceChildren(templates[index].content.cloneNode(true));
            if (focus) stations[index].focus();
        }

        stations.forEach((station, i) => {
            station.addEventListener('click', () => select(i, false));
            station.addEventListener('keydown', (event) => {
                const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
                if (event.key in keys) {
                    event.preventDefault();
                    select((i + keys[event.key] + stations.length) % stations.length, true);
                }
            });
        });
        select(0, false);
    }

    // ---------- Encoder: show the 29-number row for example travellers ----------
    function setUpEncoder() {
        const vector = document.getElementById('vector');
        const readout = document.getElementById('vector-readout');
        const columns = JSON.parse(vector.dataset.columns);

        const singleLabels = {
            form_b: ['Budget', ['$0–49', '$50–99', '$100–249', '$300+']],
            form_c: ['Season', ['winter', 'spring', 'summer', 'fall']],
            form_h: ['Pace', ['slow and easy', 'a bit of both', 'on the move all day']],
            form_i: ['Safety', ['a lot', 'somewhat', 'up for anything']],
            form_j: ['Popularity', ['off the beaten path', 'classic spots', 'mainstream and trendy']]
        };
        const multiGroups = {
            form_f: ['Experiences', { 0: 'Beach', 1: 'Adventure', 2: 'Nature', 3: 'Culture', 4: 'Nightlife', 5: 'History', 6: 'Shopping', 7: 'Food' }],
            form_g: ['Scenery', { 0: 'City', 1: 'Countryside', 2: 'Sea', 3: 'Mountains', 4: 'Lakes', 5: 'Desert', 6: 'Plains', 7: 'Jungle' }],
            form_rr: ['Continent', { a: 'Asia', c: 'Caribbean', e: 'Europe', f: 'Africa', m: 'Middle East', n: 'North America', o: 'Oceania', s: 'South America' }]
        };
        const presets = {
            beach: { form_b: 2, form_c: 2, form_h: 1, form_i: 1, form_j: 1, form_f: ['0', '7'], form_g: ['2'], form_rr: [] },
            city: { form_b: 3, form_c: 1, form_h: 1, form_i: 1, form_j: 2, form_f: ['4', '6', '7'], form_g: ['0'], form_rr: ['e'] },
            trek: { form_b: 0, form_c: 3, form_h: 2, form_i: 2, form_j: 0, form_f: ['1', '2'], form_g: ['3', '4'], form_rr: ['s', 'a'] }
        };

        // Group the columns: one group per question, in training order
        const groups = [];
        columns.forEach((column) => {
            const prefix = column in singleLabels ? column : column.replace(/_[^_]+$/, '');
            let group = groups.find((g) => g.prefix === prefix);
            if (!group) {
                const label = (singleLabels[prefix] || multiGroups[prefix])[0];
                group = { prefix, label, cells: [] };
                groups.push(group);
            }
            group.cells.push(column);
        });

        const cellElements = {};
        groups.forEach((group) => {
            const groupEl = document.createElement('div');
            groupEl.className = 'vector-group';
            const cellsEl = document.createElement('div');
            cellsEl.className = 'vector-cells';
            group.cells.forEach((column) => {
                const cell = document.createElement('span');
                cell.className = 'cell';
                cell.tabIndex = 0;
                cell.dataset.column = column;
                cell.addEventListener('pointerenter', () => describe(cell, groupEl));
                cell.addEventListener('focus', () => describe(cell, groupEl));
                cell.addEventListener('pointerleave', clearFocus);
                cell.addEventListener('blur', clearFocus);
                cellsEl.appendChild(cell);
                cellElements[column] = cell;
            });
            const labelEl = document.createElement('span');
            labelEl.className = 'vector-label';
            labelEl.textContent = group.label;
            groupEl.append(cellsEl, labelEl);
            vector.appendChild(groupEl);
        });

        function valueFor(column, preset) {
            if (column in singleLabels) return preset[column];
            const prefix = column.replace(/_[^_]+$/, '');
            const option = column.slice(prefix.length + 1);
            return preset[prefix].includes(option) ? 1 : 0;
        }

        function render(name, animate) {
            const preset = presets[name];
            let order = 0;
            columns.forEach((column) => {
                const cell = cellElements[column];
                const value = valueFor(column, preset);
                const changed = cell.textContent !== String(value);
                cell.textContent = value;
                cell.classList.toggle('is-number', column in singleLabels);
                cell.classList.toggle('is-on', !(column in singleLabels) && value === 1);
                if (animate && changed) {
                    const delay = order++ * 25;
                    setTimeout(() => {
                        cell.classList.add('is-changed');
                        setTimeout(() => cell.classList.remove('is-changed'), 250);
                    }, delay);
                }
            });
        }

        function describe(cell, groupEl) {
            const column = cell.dataset.column;
            const value = Number(cell.textContent);
            let meaning;
            if (column in singleLabels) {
                const [label, options] = singleLabels[column];
                meaning = `${label} = ${value} (${options[value]})`;
            } else {
                const prefix = column.replace(/_[^_]+$/, '');
                const [label, options] = multiGroups[prefix];
                const option = options[column.slice(prefix.length + 1)];
                meaning = `${label}: ${option} = ${value} (${value ? 'ticked' : 'not ticked'})`;
            }
            readout.innerHTML = '';
            const code = document.createElement('code');
            code.textContent = column;
            readout.append(code, ` · ${meaning}`);
            vector.classList.add('is-focused');
            vector.querySelectorAll('.vector-group').forEach((g) => g.classList.toggle('is-focused', g === groupEl));
        }

        function clearFocus() {
            vector.classList.remove('is-focused');
        }

        document.querySelectorAll('input[name="preset"]').forEach((input) => {
            input.addEventListener('change', () => render(input.value, true));
        });
        render('beach', false);
    }

    // ---------- Chart: hit rate for TripWise vs the two halves vs baselines ----------
    function setUpResultsChart(data) {
        const container = document.getElementById('results-bars');
        const title = document.getElementById('results-chart-title');
        const users = data.n_test_users;
        const rows = [
            { key: 'random_baseline', label: 'Random picks' },
            { key: 'popularity_baseline', label: 'Most popular for everyone' },
            { key: 'rf_only', label: 'Random Forest alone', ours: true },
            { key: 'et_only', label: 'Extra Trees alone', ours: true },
            { key: 'tripwise_ensemble', label: 'TripWise (both, voting)', ours: true }
        ];

        const rowElements = rows.map((row) => {
            const el = document.createElement('div');
            el.className = `bar-row${row.ours ? ' is-ours' : ''}`;
            el.tabIndex = 0;
            el.innerHTML = '<span class="bar-label"></span><span class="bar-track"><span class="bar-band"></span><span class="bar-fill"></span></span><span class="bar-value"></span>';
            el.querySelector('.bar-label').textContent = row.label;
            attachTooltip(el, () => {
                const scores = data.user_level[row.key];
                return [row.label, [
                    ['Hit rate @1', pct(scores['hit_rate@1'])],
                    ['Hit rate @5', pct(scores['hit_rate@5'])],
                    ['Hit rate @10', pct(scores['hit_rate@10'])],
                    ['F1 @10', scores['f1@10'].toFixed(3)],
                    ['NDCG @10', scores['ndcg@10'].toFixed(3)]
                ]];
            });
            container.appendChild(el);
            return el;
        });

        function draw(k) {
            const key = `hit_rate@${k}`;
            const values = rows.map((row) => data.user_level[row.key][key]);
            const max = Math.max(...values) * 100 * 1.1;
            const ours = data.user_level.tripwise_ensemble[key];
            const margin = 1.96 * Math.sqrt(ours * (1 - ours) / users);

            rowElements.forEach((el, i) => {
                el.querySelector('.bar-fill').style.setProperty('--v', (values[i] * 100 / max) * 100);
                el.querySelector('.bar-value').textContent = pct(values[i]);
                const band = el.querySelector('.bar-band');
                band.style.left = `${((ours - margin) * 100 / max) * 100}%`;
                band.style.width = `${(2 * margin * 100 / max) * 100}%`;
                el.setAttribute('aria-label', `${rows[i].label}: ${pct(values[i])}`);
            });
            title.textContent = k === 1
                ? 'Share of travellers whose top pick was a place they liked'
                : `Share of travellers with at least one yes in their top ${k}`;
        }

        document.querySelectorAll('input[name="k"]').forEach((input) => {
            input.addEventListener('change', () => draw(Number(input.value)));
        });
        draw(10);
    }

    // ---------- Chart: how much each question matters ----------
    function setUpImportanceChart() {
        const container = document.getElementById('importance-bars');
        const data = JSON.parse(container.dataset.importance);
        const names = {
            region: 'Continent', popularity: 'Popularity', budget: 'Budget', scenery: 'Scenery',
            activity: 'Pace', safety: 'Safety', season: 'Season', experiences: 'Experiences'
        };
        const entries = Object.entries(data).map(([key, { drop, std }]) => ({
            key, value: drop * 100, error: 2 * std * 100
        })).sort((a, b) => b.value - a.value);

        const min = Math.min(-1, ...entries.map((e) => e.value - e.error));
        const max = Math.max(...entries.map((e) => e.value + e.error)) * 1.08;
        const position = (v) => ((v - min) / (max - min)) * 100;
        const zero = position(0);

        entries.forEach((entry) => {
            const noise = entry.value - entry.error <= 0;
            const el = document.createElement('div');
            el.className = `bar-row is-ours${noise ? ' is-noise' : ''}`;
            el.tabIndex = 0;
            el.innerHTML = '<span class="bar-label"></span><span class="bar-track"><span class="zero-line"></span><span class="bar-fill"></span><span class="whisker"></span></span><span class="bar-value"></span>';
            el.querySelector('.bar-label').textContent = names[entry.key];
            el.querySelector('.zero-line').style.left = `${zero}%`;

            const fill = el.querySelector('.bar-fill');
            const end = position(entry.value);
            fill.style.setProperty('--start', `${Math.min(zero, end)}%`);
            fill.style.setProperty('--len', Math.abs(end - zero));
            fill.style.setProperty('--origin', entry.value >= 0 ? 'left' : 'right');
            fill.style.borderRadius = entry.value >= 0 ? '0 4px 4px 0' : '4px 0 0 4px';

            const whisker = el.querySelector('.whisker');
            whisker.style.left = `${position(entry.value - entry.error)}%`;
            whisker.style.width = `${position(entry.value + entry.error) - position(entry.value - entry.error)}%`;

            const rounded = Math.abs(entry.value).toFixed(1);
            const text = rounded === '0.0' ? '0.0' : `${entry.value >= 0 ? '+' : '−'}${rounded}`;
            el.querySelector('.bar-value').textContent = text;
            el.setAttribute('aria-label', `${names[entry.key]}: ${text} points${noise ? ', within noise' : ''}`);

            attachTooltip(el, () => [names[entry.key], [
                ['Drop in hit rate @10', `${text} pts`],
                ['Spread (±2 sd)', `±${entry.error.toFixed(1)} pts`],
                ['Verdict', noise ? 'No measurable effect' : 'Matters']
            ]]);
            container.appendChild(el);
        });
    }

    // ---------- Shared tooltip (hover and keyboard focus) ----------
    function attachTooltip(el, content) {
        const show = (x, y) => {
            const [heading, rows] = content();
            tooltip.innerHTML = '';
            const strong = document.createElement('strong');
            strong.textContent = heading;
            tooltip.appendChild(strong);
            rows.forEach(([label, value]) => {
                const line = document.createElement('span');
                const l = document.createElement('em');
                l.style.fontStyle = 'normal';
                l.textContent = label;
                const v = document.createElement('b');
                v.textContent = value;
                line.append(l, v);
                tooltip.appendChild(line);
            });
            tooltip.hidden = false;
            const box = tooltip.getBoundingClientRect();
            const left = Math.min(x + 16, window.innerWidth - box.width - 16);
            const top = y + box.height + 24 > window.innerHeight ? y - box.height - 12 : y + 16;
            tooltip.style.left = `${Math.max(16, left)}px`;
            tooltip.style.top = `${top}px`;
        };
        const hide = () => { tooltip.hidden = true; };
        el.addEventListener('pointermove', (e) => show(e.clientX, e.clientY));
        el.addEventListener('pointerleave', hide);
        el.addEventListener('focus', () => {
            const rect = el.getBoundingClientRect();
            show(rect.left + rect.width / 2, rect.bottom);
        });
        el.addEventListener('blur', hide);
    }

    function pct(value) {
        return `${(value * 100).toFixed(1)}%`;
    }
});
