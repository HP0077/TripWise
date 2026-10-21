# ==============================================================================
# TripWise - API Tests
# Run with: python -m pytest
# ==============================================================================

import pytest

import app as tripwise

FULL_PREFERENCES = {
    'budget': '2', 'season': '2', 'activity': '1', 'safety': '1', 'popularity': '1',
    'experiences': ['0', '7'], 'scenery': ['2'], 'region': []
}


@pytest.fixture
def client():
    tripwise.app.config['TESTING'] = True
    return tripwise.app.test_client()


def predict(client, payload):
    return client.post('/predict', json=payload)


def test_index_page_renders(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'id="travel-form"' in response.data


def test_predict_returns_six_ranked_destinations(client):
    body = predict(client, FULL_PREFERENCES).get_json()
    destinations = body['destinations']

    assert body['success'] is True
    assert len(destinations) == 6
    assert len({d['name'] for d in destinations}) == 6
    probabilities = [d['probability'] for d in destinations]
    assert probabilities == sorted(probabilities, reverse=True)
    assert destinations[0]['match'] == 100.0
    assert all(0 < d['match'] <= 100 for d in destinations)


@pytest.mark.parametrize('region', ['a', 'e', 'n', 's', 'f', 'o'])
def test_region_filter_only_returns_that_continent(client, region):
    body = predict(client, {**FULL_PREFERENCES, 'region': [region]}).get_json()
    assert len(body['destinations']) == 6
    assert all(tripwise.get_continent_code(d['name']) == region for d in body['destinations'])


def test_multiple_regions_are_combined(client):
    body = predict(client, {**FULL_PREFERENCES, 'region': ['f', 'o']}).get_json()
    assert all(tripwise.get_continent_code(d['name']) in ('f', 'o') for d in body['destinations'])


def test_different_preferences_give_different_results(client):
    beach = predict(client, FULL_PREFERENCES).get_json()['destinations']
    city = predict(client, {**FULL_PREFERENCES, 'budget': '3', 'experiences': ['4', '6'], 'scenery': ['0']}).get_json()['destinations']
    assert [d['name'] for d in beach] != [d['name'] for d in city]


def test_popularity_changes_the_ranking(client):
    rankings = {
        tuple(d['name'] for d in predict(client, {**FULL_PREFERENCES, 'popularity': p}).get_json()['destinations'])
        for p in ('0', '1', '2')
    }
    assert len(rankings) > 1


def test_summary_describes_the_choices(client):
    summary = predict(client, {**FULL_PREFERENCES, 'region': ['e']}).get_json()['summary']
    for expected in ('Summer', '$100-249', 'Classic spots', 'Beach', 'Food', 'Sea views', 'Europe'):
        assert expected in summary


def test_empty_preferences_still_recommend(client):
    response = predict(client, {})
    assert response.status_code == 200
    assert len(response.get_json()['destinations']) == 6


@pytest.mark.parametrize('payload', [
    {'budget': 'abc'},
    {'budget': '9'},
    {'season': -3},
    {'experiences': '0'},
    {'scenery': ['42']},
    {'region': ['x']},
])
def test_invalid_input_is_rejected(client, payload):
    response = predict(client, {**FULL_PREFERENCES, **payload})
    assert response.status_code == 400
    assert response.get_json()['success'] is False


def test_non_json_body_is_rejected(client):
    response = client.post('/predict', data='budget=2')
    assert response.status_code == 400


def test_unknown_page_returns_custom_404(client):
    response = client.get('/no-such-page')
    assert response.status_code == 404
    assert b'wrong turn' in response.data


def test_how_it_works_page_renders_metrics(client):
    response = client.get('/how-it-works')
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'Two forests, one vote' in page
    assert tripwise.page_facts['example_vote'][0]['name'].split(',')[0] in page
    assert 'F1 (macro)' in page
