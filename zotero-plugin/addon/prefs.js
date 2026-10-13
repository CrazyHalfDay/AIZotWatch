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
pref("extensions.zotwatch.llm.summaryTopN", "10");

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
