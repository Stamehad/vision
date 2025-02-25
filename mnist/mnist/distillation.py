import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, targets, temperature=2.0, alpha=0.5):
    """
    Computes knowledge distillation loss.

    student_logits: logits from the student model (shape: [batch, num_classes])
    teacher_logits: logits from the teacher model (shape: [batch, num_classes])
    targets: true labels (shape: [batch])
    temperature: softening parameter for distillation
    alpha: balance factor between soft labels and hard labels
    """

    # Compute soft labels from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    
    # Compute student's predictions as soft probabilities
    soft_preds = F.log_softmax(student_logits / temperature, dim=1)

    # KL divergence loss (soft labels)
    kd_loss = F.kl_div(soft_preds, soft_targets, reduction="batchmean") * (temperature ** 2)

    # Standard cross-entropy loss (hard labels)
    ce_loss = F.cross_entropy(student_logits, targets)

    # Weighted sum of the two losses
    return alpha * kd_loss + (1 - alpha) * ce_loss