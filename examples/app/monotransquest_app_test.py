from transquest.app.monotransquest_app import MonoTransQuestApp

test_sentences = [
        [
            "Jocurile de oferă noi provocări pentru IA în domeniul teoriei jocurilor.",
            "Games provide new challenges for IA in the area of gambling theory"
        ]
    ]

app = MonoTransQuestApp("monotransquest-da-ro_en", use_cuda=False, force_download=True)
print(app.predict_quality(test_sentences))
