import argparse
import logging
import os
import re
import numpy as np

import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from transcription_helpers import transcribe_batched

mtypes = {"cpu": "int8", "cuda": "float16"}
ENDING_PUNCTS = ".?!"
MODEL_PUNCTS = ".,;:!?"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--audio", help="name of the target audio file", required=True
    )
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation."
        "This helps with long files that don't contain a lot of music.",
    )

    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses Numerical Digits."
        "This helps the diarization accuracy but converts all digits into written text.",
    )

    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="medium.en",
        help="name of the Whisper model to use",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=8,
        help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=whisper_langs,
        help="Language spoken in the audio, specify None to perform language detection",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )
    return parser.parse_args()


def demucs_audio(args):
    """
    Process the audio file based on the provided arguments.

    This function handles the audio processing step of the diarization pipeline.
    If stemming is enabled, it uses the Demucs library to separate the vocals
    from the audio file. If stemming fails or is disabled, it returns the
    original audio file path.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The path to the processed audio file (either the separated vocals
             or the original audio file).
    """
    if args.stemming:
        try:
            # Load the audio file
            audio, sr = torchaudio.load(args.audio)

            # Ensure the audio is stereo
            if audio.shape[0] == 1:
                audio = torch.cat([audio, audio])

            # Load the Demucs model
            model = get_model("htdemucs")
            model.eval()

            # Apply the model to separate stems
            sources = apply_model(
                model, audio.unsqueeze(0), device="cpu", progress=True, num_workers=2
            )
            sources = sources.squeeze(0)

            # Extract vocals
            vocals = sources[model.sources.index("vocals")]

            # Create output path
            output_path = os.path.join(
                "temp_outputs",
                "htdemucs",
                os.path.splitext(os.path.basename(args.audio))[0],
                "vocals.wav",
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save the vocals
            torchaudio.save(output_path, vocals, sr)

            return output_path
        except Exception as e:
            logging.warning(
                f"Source splitting failed: {str(e)}. Using original audio file."
            )
            return args.audio
    return args.audio


def perform_forced_alignment(args, audio_waveform):
    alignment_model, alignment_tokenizer = load_alignment_model(
        args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    audio_waveform = (
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device)
    )
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=args.batch_size
    )

    del alignment_model
    torch.cuda.empty_cache()

    return emissions, stride, alignment_tokenizer


def process_transcript(
    full_transcript, emissions, stride, alignment_tokenizer, language
):
    tokens_starred, text_starred = preprocess_text(
        full_transcript, romanize=True, language=langs_to_iso[language]
    )
    segments, scores, blank_token = get_alignments(
        emissions, tokens_starred, alignment_tokenizer
    )
    spans = get_spans(tokens_starred, segments, blank_token)
    return postprocess_results(text_starred, spans, stride, scores)


def perform_diarization(args, audio_waveform):
    """
    Perform speaker diarization on the given audio waveform.

    This function takes the input audio waveform and performs speaker diarization
    using the NeuralDiarizer model from NVIDIA's NeMo toolkit. It handles the
    necessary preprocessing of the audio data and manages the diarization process.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration options.
        audio_waveform (numpy.ndarray or torch.Tensor): The input audio waveform.

    Returns:
        str: Path to the temporary directory containing diarization results.

    Raises:
        ValueError: If the audio_waveform has incorrect dimensions.

    Note:
        - The function creates a temporary directory to store intermediate files.
        - It converts the audio to the required format (mono, 16kHz) for diarization.
        - The NeuralDiarizer model is used for the actual diarization process.
        - GPU memory is cleared after diarization to free up resources.
    """
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)

    # Check if audio_waveform is a numpy array and convert it to a torch tensor if necessary
    if isinstance(audio_waveform, np.ndarray):
        audio_waveform = torch.from_numpy(audio_waveform)

    # Ensure the audio waveform is a 2D tensor (channels, samples)
    if audio_waveform.dim() == 1:
        audio_waveform = audio_waveform.unsqueeze(0)
    elif audio_waveform.dim() > 2:
        raise ValueError(
            "audio_waveform has too many dimensions. Expected 1D or 2D tensor."
        )

    # Move to CPU and ensure float dtype
    audio_waveform = audio_waveform.cpu().float()

    # Save the audio file
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        audio_waveform,
        16000,
        channels_first=True,
    )

    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    return temp_path


def read_speaker_timestamps(temp_path):
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        for line in f:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
    return speaker_ts


def restore_punctuation(wsm, language):
    """
    Restore punctuation to the word-speaker mapping.

    This function applies punctuation to the transcribed text using a pre-trained
    punctuation model. It works for specific languages supported by the model.

    Args:
        wsm (list): A list of dictionaries, each containing word information
                    including the word itself and its timing.
        language (str): The language code of the transcription.

    Returns:
        list: The updated word-speaker mapping with restored punctuation.

    Note:
        - The function uses the 'kredor/punctuate-all' model for punctuation.
        - If the language is not supported, it logs a warning and returns the original wsm.
        - Special handling is done for acronyms to avoid incorrect punctuation.
    """
    if language in punct_model_langs:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        words_list = list(map(lambda x: x["word"], wsm))
        labled_words = punct_model.predict(words_list, chunk_size=230)

        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ENDING_PUNCTS
                and (word[-1] not in MODEL_PUNCTS or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
    else:
        logging.warning(
            f"Punctuation restoration is not available for {language} language. Using the original punctuation."
        )

    return wsm


def main():
    args = parse_arguments()
    language = process_language_arg(args.language, args.model_name)

    vocal_target = demucs_audio(args)

    whisper_results, language, audio_waveform = transcribe_batched(
        vocal_target,
        language,
        args.batch_size,
        args.model_name,
        mtypes[args.device],
        args.suppress_numerals,
        args.device,
    )

    emissions, stride, alignment_tokenizer = perform_forced_alignment(
        args, audio_waveform
    )

    full_transcript = "".join(segment["text"] for segment in whisper_results)
    word_timestamps = process_transcript(
        full_transcript, emissions, stride, alignment_tokenizer, language
    )

    temp_path = perform_diarization(args, audio_waveform)

    speaker_ts = read_speaker_timestamps(temp_path)

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    wsm = restore_punctuation(wsm, language)

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Write output files
    with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(
        f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig"
    ) as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)


if __name__ == "__main__":
    main()
