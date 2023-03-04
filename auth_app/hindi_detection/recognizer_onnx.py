import onnx
import onnxruntime
import torch
import os
from abc import ABC, abstractmethod
import argparse
from torchvision import transforms as T
from PIL import Image
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence




class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main(image):
    model_path = os.path.join(os.getcwd(),"auth_app/hindi_detection/","parseq_resnet_hindi.onnx")
    onnx_model = onnx.load(model_path)
    node_list = []
    unicode_min = 0x0900
    unicode_max = 0x097F
    printable_glyphs = [ chr(x) for x in range(unicode_min, unicode_max+1) if chr(x).isprintable() ]
    s1 = ""
    for i in printable_glyphs:
        s1 += i
    charset  = s1
    charset = charset + "0123456789,/"

    tokenizer = Tokenizer(charset)

    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(model_path)
    img_transforms = T.Compose(([
            T.Resize((32,128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ]))

    
    # Load image and prepare for input
    image = Image.fromarray(image)
    image = img_transforms(image).unsqueeze(0)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}#, ort_session.get_inputs()[1].name: arr}
    ort_outs = ort_session.run(None, ort_inputs)[0] # numpy array
    p = torch.from_numpy(ort_outs)
    p = p.softmax(-1)
    print(p.shape)
    pred, p = tokenizer.decode(p)
    output = "".join(pred)
    return output


