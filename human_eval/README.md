### Replicating the human evaluations

This README documents our human evaluation process.
The process is identical for study1 and study2, though the scripts are slightly
different to account for the differences in the study design (the inclusion
of an "equivalent" label for annotators.)

The raw generation data for study1 is found at

        study1/sentences_test_p1.0_k0_t1.0_e0.0_h0.0006_seed1.p
        study1/sentences_test_p0.95_k0_t1.0_e0.0_h0.0_seed1.p
        
And similary for study2 (though we use seed 2 for study2.) All the study2 file names
are slightly different to indicate they were for study2, but the process is identical.

To pick the prefixes that are (1) long enough for the study, and (2) pass _your_
human filter of being a reasonable natural language prompt, run

        make_test_jsonl.py 
        
You can look at the decisions we made for filtering, in `for_human_eval-test.jsonl`, and
`rejected_for_human_eval-test.jsonl`. The script is interactive; you provide a 1 to keep
a prompt and a 0 to filter it. 

To turn the jsonl of chosen documents into a CSV for mturk, run

        jsonl_to_csv_test.py for_human_eval-test.jsonl
        
which will generate a csv titled `for_human_eval-test.jsonl.csv`, which can be uploaded
to mturk. This randomizes the order in which the generations of each method are presented
in the user interface, to avoid biases about picking a particular (like the first.) 
(It also removes disallowed characters, etc.). It stores the name of the method corresponding
to each option in the CSV for recovery later.

When you get the responses back from mturk, you'll get a batch file. To convert this
back to jsonl (and obfuscate the actual unique worker id for an identifer
local to this study), run

        strip_identifiables_for_publishing.py [mturk_batch_file]

For our results, this resulted in our `study1_results.json`, which you can look at.
To then summarize the results (resolve the randomly ordered labels to their method names
and report the counts of votes for each method), run

        report-test.py

