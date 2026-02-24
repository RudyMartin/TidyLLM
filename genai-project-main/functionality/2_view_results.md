You're absolutely right! When clicking "view" on a processed document, users should see the compliance analysis results. Currently, the system sets the currentDocument but doesn't generate/load the compliance scores, so the results screen doesn't appear.

I need to:
Generate compliance scores when a document is selected for viewing
Load the scores into the complianceScores state 
Set up the proper review scope from the document's configuration
Ensure the ComplianceResults component displays the analysis results

This will make the "view" action actually show the document's compliance analysis results as expected
