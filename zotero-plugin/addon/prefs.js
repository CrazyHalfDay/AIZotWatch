// Default preferences. Numbers are stored as strings (bound to text inputs).
pref("extensions.zotwatch.embedding.provider", "local");
pref("extensions.zotwatch.embedding.apiKey", "");
pref("extensions.zotwatch.embedding.model", "");
pref("extensions.zotwatch.embedding.baseURL", "");
pref("extensions.zotwatch.embedding.dimensions", "1024");
pref("extensions.zotwatch.embedding.batchSize", "");
pref("extensions.zotwatch.embedding.concurrency", "3");

pref("extensions.zotwatch.llm.enabled", false);
pref("extensions.zotwatch.llm.baseURL", "https://openrouter.ai/api/v1");
pref("extensions.zotwatch.llm.apiKey", "");
pref("extensions.zotwatch.llm.model", "deepseek/deepseek-chat");
pref("extensions.zotwatch.llm.translateTitles", true);
pref("extensions.zotwatch.llm.topicLabels", true);
pref("extensions.zotwatch.llm.summaryTopN", "10");
pref("extensions.zotwatch.llm.backup.baseURL", "");
pref("extensions.zotwatch.llm.backup.apiKey", "");
pref("extensions.zotwatch.llm.backup.model", "");

pref("extensions.zotwatch.sources.crossref.enabled", true);
pref("extensions.zotwatch.sources.crossref.mailto", "");
pref("extensions.zotwatch.sources.crossref.daysBack", "7");
pref("extensions.zotwatch.sources.crossref.maxResults", "3000");
pref("extensions.zotwatch.sources.arxiv.enabled", false);
pref("extensions.zotwatch.sources.arxiv.categories", "physics.geo-ph, astro-ph.EP");
pref("extensions.zotwatch.sources.arxiv.daysBack", "7");
pref("extensions.zotwatch.sources.arxiv.maxResults", "500");
pref("extensions.zotwatch.sources.eartharxiv.enabled", false);
pref("extensions.zotwatch.sources.eartharxiv.daysBack", "14");
pref("extensions.zotwatch.sources.eartharxiv.maxResults", "500");
pref("extensions.zotwatch.sources.openalexAbstracts", true);
pref("extensions.zotwatch.abstracts.crossref", true);
pref("extensions.zotwatch.abstracts.elsevierKey", "");
pref("extensions.zotwatch.abstracts.springerKey", "");

// Scholarly data APIs (学术数据接口); the Crossref contact address stays sources.crossref.mailto
pref("extensions.zotwatch.apis.openalexKey", "");
pref("extensions.zotwatch.apis.semanticScholarKey", "");

pref("extensions.zotwatch.followed.enabled", false);
pref("extensions.zotwatch.followed.authors", "");
pref("extensions.zotwatch.followed.daysBack", "60");

// Flagship geoscience track; empty thresholds are calibrated per embedding provider
pref("extensions.zotwatch.flagship.enabled", false);
pref("extensions.zotwatch.flagship.issns", "0028-0836\n1476-4687\n0036-8075\n1095-9203\n2041-1723\n2375-2548\n2662-4435\n1091-6490\n0027-8424\n1752-0894\n1752-0908\n2662-138X");
pref("extensions.zotwatch.flagship.positiveAnchor", "Solid Earth geoscience: igneous, metamorphic and sedimentary petrology, geochemistry and isotope geochemistry, mineralogy, tectonics and structural geology, geochronology, ore deposits and economic geology, volcanology, the deep Earth, stratigraphy, and paleontology / paleobiology.");
pref("extensions.zotwatch.flagship.negativeAnchors", "Atmospheric science, meteorology, climate dynamics, air quality, weather and atmospheric circulation.\nCondensed matter and materials physics: ferroelectrics, semiconductors, nanomaterials, thin films, crystal device structures, electron microscopy of engineered materials, quantum, topological and magnetic materials.\nChemistry: synthesis, catalysis, electrochemistry, batteries and energy storage, polymers, molecular, organic and analytical chemistry.\nBiology and biomedicine: genetics and genomics, molecular and cell biology, neuroscience, physiology, immunology, microbiology, ecology and evolution.\nAstronomy, astrophysics, cosmology, exoplanets and the atmospheres of other planets.\nEngineering, computer science, machine learning, optics, photonics and applied device physics.");
pref("extensions.zotwatch.flagship.minScore", "");
pref("extensions.zotwatch.flagship.grayLow", "");
pref("extensions.zotwatch.flagship.llmFallback", true);
pref("extensions.zotwatch.flagship.llmBoundary", "ONLY solid-earth geoscience or paleontology (igneous/metamorphic/sedimentary petrology, geochemistry and isotope geochemistry, mineralogy, tectonics and structural geology, geochronology, ore deposits, volcanology, stratigraphy, paleontology/paleobiology, the deep Earth). EXCLUDE everything else, including materials science and condensed-matter physics, chemistry, biology and biomedicine, astronomy and astrophysics, atmospheric science/meteorology/climate, engineering and computer science — even when published in a general-science journal.");
pref("extensions.zotwatch.flagship.maxResults", "30");

pref("extensions.zotwatch.journals.whitelist", "");
pref("extensions.zotwatch.journals.autoFromLibrary", true);
pref("extensions.zotwatch.journals.autoTopN", "40");
pref("extensions.zotwatch.journals.autoMinItems", "2");

pref("extensions.zotwatch.scoring.topK", "30");
pref("extensions.zotwatch.scoring.recentDays", "7");
pref("extensions.zotwatch.scoring.maxPreprintRatio", "0.7");
pref("extensions.zotwatch.scoring.requireAbstract", true);
pref("extensions.zotwatch.scoring.minSimilarity", "");
pref("extensions.zotwatch.scoring.includeKeywords", "");
pref("extensions.zotwatch.scoring.excludeKeywords", "");
pref("extensions.zotwatch.scoring.thresholdMode", "dynamic");
pref("extensions.zotwatch.scoring.temporalHalflifeDays", "0");

pref("extensions.zotwatch.output.collection", "ZotWatch 推荐");
pref("extensions.zotwatch.output.tag", "ZotWatch");
pref("extensions.zotwatch.output.summaryNote", true);
pref("extensions.zotwatch.output.useTranslator", true);

pref("extensions.zotwatch.schedule.autoRun", true);
pref("extensions.zotwatch.schedule.intervalHours", "24");
pref("extensions.zotwatch.schedule.lastRun", "");
pref("extensions.zotwatch.schedule.lastAttempt", "");

pref("extensions.zotwatch.inbox.retentionDays", "30");
pref("extensions.zotwatch.profile.libraryID", "");
