document.addEventListener("DOMContentLoaded", () => {
  const prefillBtn = document.getElementById("btn-prefill");
  if (!prefillBtn) return;

  prefillBtn.addEventListener("click", () => {
    // Very rough example values; update to your liking
    const example = {
      latitude: 37.77,
      longitude: -122.42,
      age_first_funding_year: 1.2,
      age_last_funding_year: 3.0,
      age_first_milestone_year: 1.5,
      age_last_milestone_year: 2.2,
      relationships: 5,
      funding_rounds: 2,
      funding_total_usd: 5000000,
      milestones: 2,
      category_code: 30, // will be ignored if category_code_text is set and encoder present
      has_VC: 1,
      has_angel: 0,
      has_roundA: 1,
      has_roundB: 0,
      has_roundC: 0,
      has_roundD: 0,
      avg_participants: 3.0,
      is_top500: 1,
      founded_at_year: 2009,
      founded_at_day: 1,
      founded_at_month: 1,
      first_funding_at_year: 2010,
      first_funding_at_day: 1,
      first_funding_at_month: 4,
      last_funding_at_year: 2012,
      last_funding_at_day: 1,
      last_funding_at_month: 4,
      closed_at: 0
    };

    Object.entries(example).forEach(([k, v]) => {
      const el = document.querySelector(`[name="${k}"]`);
      if (el) el.value = v;
    });
  });
});
