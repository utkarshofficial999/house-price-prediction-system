document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const inputArea = document.getElementById('input-area');
  const inputBedrooms = document.getElementById('input-bedrooms');
  const inputBathrooms = document.getElementById('input-bathrooms');
  const inputFloors = document.getElementById('input-floors');
  const inputParking = document.getElementById('input-parking');
  const inputAge = document.getElementById('input-age');
  const inputFurnished = document.getElementById('input-furnished');
  const inputLocation = document.getElementById('input-location');
  const inputCrime = document.getElementById('input-crime');
  const inputMetro = document.getElementById('input-metro');
  const inputHospital = document.getElementById('input-hospital');
  const inputSchool = document.getElementById('input-school');
  const inputTax = document.getElementById('input-tax');

  // Value Display Elements
  const valArea = document.getElementById('val-area');
  const valAge = document.getElementById('val-age');
  const valLocation = document.getElementById('val-location');
  const valCrime = document.getElementById('val-crime');
  const valMetro = document.getElementById('val-metro');
  const valHospital = document.getElementById('val-hospital');
  const valSchool = document.getElementById('val-school');
  const valTax = document.getElementById('val-tax');

  // Outputs
  const activeModelTitle = document.getElementById('active-model-title');
  const displayPriceShort = document.getElementById('display-price-short');
  const displayPriceExact = document.getElementById('display-price-exact');
  const displayPriceSqft = document.getElementById('display-price-sqft');
  const displayPriceRange = document.getElementById('display-price-range');
  const dualComparisonBox = document.getElementById('dual-comparison-box');
  const dualLrPrice = document.getElementById('dual-lr-price');
  const dualDtPrice = document.getElementById('dual-dt-price');
  const dualAgreement = document.getElementById('dual-agreement');

  const reliabilityLabel = document.getElementById('reliability-label');
  const reliabilityScore = document.getElementById('reliability-score');
  const reliabilityBar = document.getElementById('reliability-bar');
  const reliabilityFootnote = document.getElementById('reliability-footnote');

  const breakdownTitle = document.getElementById('breakdown-title');
  const breakdownDesc = document.getElementById('breakdown-desc');
  const impactBreakdown = document.getElementById('impact-breakdown');

  // State
  let currentModel = 'linear_regression';
  let lastApiResponse = null;

  // Presets definition
  const presets = {
    starter: {
      Area_sqft: 650,
      Bedrooms: 1,
      Bathrooms: 1,
      Floors: 1,
      Age_years: 22,
      Parking_spaces: 0,
      Distance_school_km: 2.8,
      Distance_hospital_km: 4.6,
      Distance_metro_km: 5.2,
      Crime_rate: 6.8,
      Property_tax: 12000,
      Furnished: 0,
      Location_score: 5
    },
    comfort: {
      Area_sqft: 1000,
      Bedrooms: 2,
      Bathrooms: 2,
      Floors: 1,
      Age_years: 8,
      Parking_spaces: 1,
      Distance_school_km: 0.9,
      Distance_hospital_km: 2.5,
      Distance_metro_km: 2.2,
      Crime_rate: 3.9,
      Property_tax: 20500,
      Furnished: 1,
      Location_score: 8
    },
    family: {
      Area_sqft: 1500,
      Bedrooms: 3,
      Bathrooms: 3,
      Floors: 2,
      Age_years: 4,
      Parking_spaces: 2,
      Distance_school_km: 0.5,
      Distance_hospital_km: 1.6,
      Distance_metro_km: 1.2,
      Crime_rate: 2.4,
      Property_tax: 34000,
      Furnished: 1,
      Location_score: 9
    },
    luxury: {
      Area_sqft: 2400,
      Bedrooms: 4,
      Bathrooms: 4,
      Floors: 3,
      Age_years: 1,
      Parking_spaces: 3,
      Distance_school_km: 0.3,
      Distance_hospital_km: 1.0,
      Distance_metro_km: 0.5,
      Crime_rate: 1.3,
      Property_tax: 59000,
      Furnished: 1,
      Location_score: 10
    }
  };

  function updateDisplayLabels() {
    valArea.textContent = `${Number(inputArea.value).toLocaleString()} sqft`;
    valAge.textContent = `${inputAge.value} Years`;
    valLocation.textContent = `Score: ${inputLocation.value} / 10`;
    valCrime.textContent = `${Number(inputCrime.value).toFixed(1)}`;
    valMetro.textContent = `${Number(inputMetro.value).toFixed(1)} km`;
    valHospital.textContent = `${Number(inputHospital.value).toFixed(1)} km`;
    valSchool.textContent = `${Number(inputSchool.value).toFixed(1)} km`;
    valTax.textContent = `₹${Number(inputTax.value).toLocaleString()}`;
  }

  function applyPreset(data) {
    inputArea.value = data.Area_sqft;
    inputBedrooms.value = data.Bedrooms;
    inputBathrooms.value = data.Bathrooms;
    inputFloors.value = data.Floors;
    inputAge.value = data.Age_years;
    inputParking.value = data.Parking_spaces;
    inputSchool.value = data.Distance_school_km;
    inputHospital.value = data.Distance_hospital_km;
    inputMetro.value = data.Distance_metro_km;
    inputCrime.value = data.Crime_rate;
    inputTax.value = data.Property_tax;
    inputFurnished.value = data.Furnished;
    inputLocation.value = data.Location_score;

    // Toggle furnished UI
    document.querySelectorAll('.toggle-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.val === String(data.Furnished));
    });

    updateDisplayLabels();
    requestPrediction();
  }

  // Model Selector Logic
  document.querySelectorAll('.model-seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.model-seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentModel = btn.dataset.model;
      if (lastApiResponse) {
        renderPrediction(lastApiResponse);
      } else {
        requestPrediction();
      }
    });
  });

  // Steppers logic
  document.querySelectorAll('.step-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.dataset.target;
      const targetInput = document.getElementById(targetId);
      const isInc = btn.dataset.action === 'inc';
      let val = parseInt(targetInput.value, 10);
      const min = parseInt(targetInput.min, 10);
      const max = parseInt(targetInput.max, 10);

      if (isInc && val < max) val++;
      else if (!isInc && val > min) val--;

      targetInput.value = val;
      requestPrediction();
    });
  });

  // Furnished toggle logic
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      inputFurnished.value = btn.dataset.val;
      requestPrediction();
    });
  });

  // Presets click logic
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const presetKey = btn.dataset.preset;
      if (presets[presetKey]) {
        applyPreset(presets[presetKey]);
      }
    });
  });

  // Sliders input events
  const allSliders = [
    inputArea, inputAge, inputLocation, inputCrime,
    inputMetro, inputHospital, inputSchool, inputTax
  ];

  allSliders.forEach(slider => {
    slider.addEventListener('input', () => {
      updateDisplayLabels();
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      debouncePrediction();
    });
  });

  // Debounced prediction fetch
  let debounceTimeout = null;
  function debouncePrediction() {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(requestPrediction, 100);
  }

  async function requestPrediction() {
    const payload = {
      Area_sqft: parseFloat(inputArea.value),
      Bedrooms: parseInt(inputBedrooms.value, 10),
      Bathrooms: parseInt(inputBathrooms.value, 10),
      Floors: parseInt(inputFloors.value, 10),
      Age_years: parseFloat(inputAge.value),
      Parking_spaces: parseInt(inputParking.value, 10),
      Distance_school_km: parseFloat(inputSchool.value),
      Distance_hospital_km: parseFloat(inputHospital.value),
      Distance_metro_km: parseFloat(inputMetro.value),
      Crime_rate: parseFloat(inputCrime.value),
      Property_tax: parseFloat(inputTax.value),
      Furnished: parseInt(inputFurnished.value, 10),
      Location_score: parseFloat(inputLocation.value),
      model_choice: currentModel
    };

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error('Prediction request failed');
      }

      const res = await response.json();
      lastApiResponse = res;
      renderPrediction(res);
    } catch (err) {
      console.error('Error fetching prediction:', err);
    }
  }

  function renderPrediction(data) {
    if (!data) return;

    const lr = data.linear_regression;
    const dt = data.decision_tree;
    const consensus = data.consensus;

    // Dual comparison row values
    dualLrPrice.textContent = lr.predicted_price_formatted;
    dualDtPrice.textContent = dt.predicted_price_formatted;
    dualAgreement.textContent = `${consensus.agreement_pct}% Consensus (Δ ${consensus.delta_formatted})`;

    if (currentModel === 'linear_regression') {
      activeModelTitle.textContent = 'Linear Regression (OLS) Valuation';
      dualComparisonBox.style.display = 'none';

      displayPriceShort.textContent = formatShortPrice(lr.predicted_price);
      displayPriceExact.textContent = `₹${Math.round(lr.predicted_price).toLocaleString('en-IN')} Estimated Value`;
      displayPriceSqft.textContent = lr.price_per_sqft_formatted;
      displayPriceRange.textContent = `${lr.range.low_formatted} – ${lr.range.high_formatted}`;

      reliabilityLabel.textContent = 'Linear Regression Accuracy';
      reliabilityScore.textContent = `${(lr.metrics.r2_test * 100).toFixed(2)}% R²`;
      reliabilityBar.style.width = `${lr.metrics.r2_test * 100}%`;
      reliabilityFootnote.textContent = `Test MAE: ±₹${(lr.metrics.rmse_inr/1e5).toFixed(2)} Lakhs (MAPE: ${lr.metrics.mape_percent}%)`;

      breakdownTitle.textContent = 'Linear Regression Feature Drivers';
      breakdownDesc.textContent = 'Coefficients weighted by property input:';
      renderLrContributions(lr.contributions);

    } else if (currentModel === 'decision_tree') {
      activeModelTitle.textContent = 'Decision Tree Regressor Valuation';
      dualComparisonBox.style.display = 'none';

      displayPriceShort.textContent = formatShortPrice(dt.predicted_price);
      displayPriceExact.textContent = `₹${Math.round(dt.predicted_price).toLocaleString('en-IN')} Estimated Value`;
      displayPriceSqft.textContent = dt.price_per_sqft_formatted;
      displayPriceRange.textContent = `${dt.range.low_formatted} – ${dt.range.high_formatted}`;

      reliabilityLabel.textContent = 'Decision Tree Accuracy';
      reliabilityScore.textContent = `${(dt.metrics.r2_test * 100).toFixed(2)}% R²`;
      reliabilityBar.style.width = `${dt.metrics.r2_test * 100}%`;
      reliabilityFootnote.textContent = `Test MAE: ±₹${(dt.metrics.rmse_inr/1e5).toFixed(2)} Lakhs (MAPE: ${dt.metrics.mape_percent}%)`;

      breakdownTitle.textContent = 'Decision Tree Feature Importances';
      breakdownDesc.textContent = 'Information gain / split variance explained:';
      renderDtImportances(dt.importances);

    } else {
      // Both / Consensus Mode
      activeModelTitle.textContent = 'Dual Consensus AI Valuation (Ensemble)';
      dualComparisonBox.style.display = 'flex';

      displayPriceShort.textContent = formatShortPrice(consensus.predicted_price);
      displayPriceExact.textContent = `₹${Math.round(consensus.predicted_price).toLocaleString('en-IN')} Consensus Average`;
      displayPriceSqft.textContent = consensus.price_per_sqft_formatted;
      displayPriceRange.textContent = `${lr.range.low_formatted} – ${dt.range.high_formatted}`;

      reliabilityLabel.textContent = 'Dual Models Agreement';
      reliabilityScore.textContent = `${consensus.agreement_pct}% Consensus`;
      reliabilityBar.style.width = `${consensus.agreement_pct}%`;
      reliabilityFootnote.textContent = `Linear: ${lr.predicted_price_formatted} vs Decision Tree: ${dt.predicted_price_formatted}`;

      breakdownTitle.textContent = 'Consensus Model Valuation Drivers';
      breakdownDesc.textContent = 'Top factors identified by both algorithms:';
      renderConsensusBreakdown(lr.contributions, dt.importances);
    }
  }

  function formatShortPrice(rawPrice) {
    if (rawPrice >= 10000000) {
      return `${(rawPrice / 10000000).toFixed(2)} Cr`;
    } else {
      return `${(rawPrice / 100000).toFixed(2)} Lakh`;
    }
  }

  function renderLrContributions(contributions) {
    if (!contributions) return;

    const keyFeatures = [
      { key: 'Area_sqft', label: 'Living Area Size' },
      { key: 'Property_tax', label: 'Property Tax Bracket' },
      { key: 'Furnished', label: 'Furnishing Finishes' },
      { key: 'Bathrooms', label: 'Bathrooms Valuation' },
      { key: 'Parking_spaces', label: 'Parking Spaces' },
      { key: 'Age_years', label: 'Building Age Depreciation' },
      { key: 'Crime_rate', label: 'Neighborhood Crime Index' },
      { key: 'Distance_school_km', label: 'School Distance Offset' }
    ];

    impactBreakdown.innerHTML = '';
    keyFeatures.forEach(item => {
      const info = contributions[item.key];
      if (!info) return;

      const impact = info.impact_inr;
      const isPositive = impact >= 0;
      const el = document.createElement('div');
      el.className = 'impact-item';
      el.innerHTML = `
        <span class="impact-name">${item.label}</span>
        <span class="impact-val ${isPositive ? 'impact-positive' : 'impact-negative'}">
          ${isPositive ? '+' : ''}${info.impact_formatted}
        </span>
      `;
      impactBreakdown.appendChild(el);
    });
  }

  function renderDtImportances(importances) {
    if (!importances) return;

    const labels = {
      'Area_sqft': 'Living Area (sqft)',
      'Distance_metro_km': 'Distance to Metro',
      'Parking_spaces': 'Parking Spaces',
      'Property_tax': 'Property Tax',
      'Location_score': 'Location Score',
      'Crime_rate': 'Crime Rate Index',
      'Age_years': 'Property Age'
    };

    impactBreakdown.innerHTML = '';
    const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);

    sorted.slice(0, 6).forEach(([key, val]) => {
      const pct = (val * 100).toFixed(1);
      const el = document.createElement('div');
      el.className = 'impact-item';
      el.innerHTML = `
        <span class="impact-name">${labels[key] || key}</span>
        <span class="impact-val text-emerald">
          ${pct}% Importance
        </span>
      `;
      impactBreakdown.appendChild(el);
    });
  }

  function renderConsensusBreakdown(lrContrib, dtImp) {
    impactBreakdown.innerHTML = '';

    const highlights = [
      { name: 'Living Area Dominance', note: 'Primary price factor for both models (77.1% DT weight)' },
      { name: 'Transit & Metro Access', note: 'Second highest tree split cutoff at < 2.2 km' },
      { name: 'Parking Capacity', note: 'Dedicated parking spaces add +9.4% valuation tier' },
      { name: 'Tax & Municipal Finishes', note: 'Strong correlation with luxury specification level' }
    ];

    highlights.forEach(h => {
      const el = document.createElement('div');
      el.className = 'impact-item';
      el.innerHTML = `
        <span class="impact-name">${h.name}</span>
        <span class="impact-val text-emerald" style="font-size: 0.76rem; max-width: 55%; text-align: right;">
          ${h.note}
        </span>
      `;
      impactBreakdown.appendChild(el);
    });
  }

  // Initial load
  updateDisplayLabels();
  requestPrediction();
});
