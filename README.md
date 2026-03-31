# TITANIC-SIM
AI-driven historical survival simulator with LLM narrative and probabilistic outcome modeling
An LLM acts as a "Story Master" it places you aboard the RMS Titanic in 1912. Period-authentic dialogue, real social dynamics, the tension of that night. As you make decisions, it quietly collects information about you: your class, your gender, your family around you, how much you paid for your ticket.

Then it passes that data structured as JSON to a FastAPI backend running a trained logistic regression model.

The model returns a verdict

