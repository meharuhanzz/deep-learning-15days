"""Day 11 hands-on: load a REAL pretrained transformer (bert-tiny, ~17MB -
chosen specifically for its small size, since this machine runs close to
its disk quota; see ../../deeplearning root README for that constraint)
via Hugging Face `transformers`, inspect its tokenizer, and fine-tune a
small classification head on top of it for a toy sentiment task.

This mirrors Day 8's transfer-learning pattern exactly (freeze/adapt a
pretrained backbone, train a new head) - just for text instead of images.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

torch.manual_seed(0)
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"  # Google-maintained tiny BERT: 2 layers, 128 hidden, ~4.4M params, ~17MB


def make_toy_sentiment_data(n_per_class=150, seed=0):
    """A tiny, fully synthetic sentiment dataset (no external data file) -
    positive/negative template sentences built from small word banks, so
    the "true" label is unambiguous and mechanically checkable."""
    import random
    rng = random.Random(seed)
    subjects = ["the movie", "this restaurant", "the service", "that book", "the trip", "her performance"]
    pos_words = ["excellent", "wonderful", "fantastic", "delightful", "impressive", "outstanding"]
    neg_words = ["terrible", "awful", "disappointing", "mediocre", "frustrating", "boring"]
    templates_pos = ["{s} was {w}.", "I thought {s} was {w} overall.", "Honestly, {s} felt {w}."]
    templates_neg = templates_pos

    texts, labels = [], []
    for _ in range(n_per_class):
        s, w, t = rng.choice(subjects), rng.choice(pos_words), rng.choice(templates_pos)
        texts.append(t.format(s=s.capitalize() if t.startswith("{s}") else s, w=w))
        labels.append(1)
    for _ in range(n_per_class):
        s, w, t = rng.choice(subjects), rng.choice(neg_words), rng.choice(templates_neg)
        texts.append(t.format(s=s.capitalize() if t.startswith("{s}") else s, w=w))
        labels.append(0)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), torch.tensor(labels)


class BertTinyClassifier(nn.Module):
    def __init__(self, n_classes=2, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = out.last_hidden_state[:, 0, :]  # the [CLS] token's final representation
        return self.head(cls_hidden)


def demo_tokenizer(tokenizer):
    """Subword tokenization: an unfamiliar/rare word gets split into
    smaller known PIECES rather than becoming a single <UNK> token -
    this is WordPiece (BERT's scheme; GPT-family models use byte-pair
    encoding, a closely related idea) and it's WHY transformer vocabularies
    can stay a manageable size (bert-tiny: ~30k tokens) while still
    handling words never seen during training."""
    examples = ["deep learning", "unbelievably", "supercalifragilisticexpialidocious"]
    print("=== subword tokenization examples ===")
    for text in examples:
        tokens = tokenizer.tokenize(text)
        print(f"  {text!r:45s} -> {tokens}")
    enc = tokenizer("the movie was excellent.", return_tensors="pt")
    print("\n=== full encoding (what actually feeds the model) ===")
    print("  input_ids:     ", enc["input_ids"])
    print("  attention_mask:", enc["attention_mask"])
    print("  decoded back:  ", tokenizer.decode(enc["input_ids"][0]))


def main():
    print(f"loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # bert-tiny shares this vocab; bert-base-uncased ships a modern fast tokenizer.json (small: vocab+config only, not its model weights)
    demo_tokenizer(tokenizer)

    texts, labels = make_toy_sentiment_data()
    split = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split], labels[:split]
    val_texts, val_labels = texts[split:], labels[split:]

    def encode(batch_texts):
        return tokenizer(list(batch_texts), padding=True, truncation=True,
                          max_length=32, return_tensors="pt")

    print("\n=== fine-tuning a classification head on frozen bert-tiny ===")
    model = BertTinyClassifier(freeze_backbone=True)
    opt = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(15):
        model.train()
        perm = torch.randperm(len(train_texts))
        for i in range(0, len(train_texts), 16):
            idx = perm[i:i + 16].tolist()
            batch_texts = [train_texts[j] for j in idx]
            batch_labels = train_labels[torch.tensor(idx)]
            enc = encode(batch_texts)
            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = loss_fn(logits, batch_labels)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            enc = encode(val_texts)
            val_logits = model(enc["input_ids"], enc["attention_mask"])
            val_acc = (val_logits.argmax(-1) == val_labels).float().mean().item()
        if epoch % 3 == 0 or epoch == 14:
            print(f"  epoch {epoch:2d}  val_acc {val_acc:.3f}")

    # a couple of hand-written sanity-check sentences, never seen in training
    model.eval()
    test_sentences = ["The staff were incredibly rude and slow.",
                       "What a fantastic, memorable evening."]
    with torch.no_grad():
        enc = encode(test_sentences)
        preds = model(enc["input_ids"], enc["attention_mask"]).argmax(-1)
    print("\n=== held-out hand-written sentences ===")
    for s, p in zip(test_sentences, preds):
        print(f"  {'positive' if p.item()==1 else 'negative':9s}  {s!r}")


if __name__ == "__main__":
    main()
