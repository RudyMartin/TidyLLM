ll 9 demo apps are now complete and exported. Here's a summary:

  Summary: 9 YRSN Demo Apps Created

  Tier 1 - Core YRSN Capabilities

  | App                 | Tools                                                                              | Key Feature                               |
  |---------------------|------------------------------------------------------------------------------------|-------------------------------------------|
  | rag_optimizer_app   | score_chunk, optimize_rag, analyze_rag_quality, compare_rag_strategies             | RAG result optimization (35%→71% quality) |
  | prompt_clinic_app   | analyze_prompt_section, diagnose_prompt, improve_prompt, compare_prompts           | Prompt quality grading (A-F scale)        |
  | context_trimmer_app | analyze_sentences, trim_context, smart_compress, extract_key_content, remove_noise | Context compression (47%→100% quality)    |

  Tier 2 - Information Management

  | App                 | Tools                                                                                         | Key Feature                   |
  |---------------------|-----------------------------------------------------------------------------------------------|-------------------------------|
  | search_reranker_app | score_result, rerank_results, compare_rankings, diversify_results                             | Source-weighted reranking     |
  | chat_memory_app     | score_message, optimize_history, summarize_history, detect_context_bloat                      | Recency decay + pruning       |
  | doc_classifier_app  | score_document, classify_documents, rank_documents, find_best_documents, analyze_document_set | Relevance tier classification |

  Tier 3 - Advanced

  | App                        | Tools                                                                                         | Key Feature                             |
  |----------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------|
  | hallucination_detector_app | check_response, compare_to_sources, detect_collapse_type, analyze_confidence                  | Grounding + collapse detection          |
  | multi_source_fusion_app    | analyze_source, detect_conflicts, fuse_sources, reconcile_conflicts, compute_source_agreement | Conflict detection (CONFUSION collapse) |
  | adaptive_router_app        | analyze_request, route_request, compute_temperature, batch_route, estimate_cost               | τ=1/α routing (38.8% cost savings)      |

  All apps are now accessible via:
  from yrsn_context.apps import rag_optimizer_app, prompt_clinic_app, ...
