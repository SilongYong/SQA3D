# utils.sqa_data_2_ClipBERT args.output_dir needs to be fixed
# other than that, utils is done
# now working on ScanQA

# # NOTE: bert-based -> remove (DONE)
# # NOTE: use_lang_cls -> use_aux_reg(lang_classifier -> aux_regressor)    when load_ckpt, key(self.lang_cls) -> key(self.aux_reg)   (DONE)
# # NOTE: data_dict "lang_scores" -> "aux_scores"  (DONE)
# # NOTE: remove use_reference    (ref_iou_rate_0)? (ref in eval_helper)?    (DONE)
# # NOTE: remove no vote     (DONE)
# # NOTE: remove sam   ([aug] in sepdataset.py)  (DONE)
# # NOTE: remove "aug" in train_new.py           (DONE)
# # NOTE: remove small                           (DONE)
# # NOTE: remove stage_test                      (DONE)
# # NOTE: remove adv_cls    answer_wrong         (DONE)
# # NOTE: remove seg                             (DONE)
# # NOTE: remove no_fix_answer                   (DONE)
# # NOTE: remove use_gt                          (DONE)
# # NOTE: TODO: check if test in sepdataset.py needs to be deleted
# # NOTE: load checkpoint using strict=False      (ALMOST DONE)
# # NOTE: test_new.py

# next step : mcan
# # NOTE: load_data.py      (DONE)
# # NOTE: train_gqa.py      (DONE) (ALMOST?)

# next step : ClipBERT
# # NOTE: How to generate data?
# # NOTE: 

# next step : README.md
# TODO: (answer_counter.json) (DONE)