import os
import json
import shutil
from typing import Dict, List, Optional, Union, BinaryIO
from datetime import datetime
import wave
import soundfile as sf
from .logger import default_logger as logger

class FileHandler:
    """Handles file operations for the AI detection system."""
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the file handler.
        
        Args:
            base_dir: Base directory for file storage
        """
        self.base_dir = base_dir
        self.audio_dir = os.path.join(base_dir, "audio")
        self.transcript_dir = os.path.join(base_dir, "transcripts")
        self.results_dir = os.path.join(base_dir, "results")
        
        # Create necessary directories
        self._create_directories()
        
        # Supported audio formats
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.flac'}

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.base_dir, self.audio_dir, self.transcript_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
            logger.log_file_operation("directory creation", directory, True)

    def save_audio_file(self, file: Union[str, BinaryIO], filename: Optional[str] = None) -> str:
        """
        Save an audio file to the audio directory.
        
        Args:
            file: Path to audio file or file-like object
            filename: Optional filename to use (generated if not provided)
            
        Returns:
            str: Path to saved audio file
        """
        try:
            if isinstance(file, str):
                # If file is a path
                ext = os.path.splitext(file)[1].lower()
                if ext not in self.supported_formats:
                    raise ValueError(f"Unsupported audio format: {ext}")
                
                if not filename:
                    filename = os.path.basename(file)
                
                dest_path = os.path.join(self.audio_dir, filename)
                shutil.copy2(file, dest_path)
            else:
                # If file is a file-like object
                if not filename:
                    filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                
                dest_path = os.path.join(self.audio_dir, filename)
                with open(dest_path, 'wb') as f:
                    shutil.copyfileobj(file, f)
            
            logger.log_file_operation("save", dest_path, True)
            return dest_path
            
        except Exception as e:
            logger.log_error(f"Error saving audio file {filename}", e)
            raise

    def save_transcript(self, transcript: str, file_id: str) -> str:
        """
        Save a transcript to the transcript directory.
        
        Args:
            transcript: Transcript text
            file_id: Associated file ID
            
        Returns:
            str: Path to saved transcript file
        """
        try:
            filename = f"{file_id}_transcript.txt"
            file_path = os.path.join(self.transcript_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            logger.log_file_operation("save", file_path, True)
            return file_path
            
        except Exception as e:
            logger.log_error(f"Error saving transcript for {file_id}", e)
            raise

    def save_analysis_results(self, results: Dict, file_id: str) -> str:
        """
        Save analysis results to the results directory.
        
        Args:
            results: Analysis results dictionary
            file_id: Associated file ID
            
        Returns:
            str: Path to saved results file
        """
        try:
            filename = f"{file_id}_results.json"
            file_path = os.path.join(self.results_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            logger.log_file_operation("save", file_path, True)
            return file_path
            
        except Exception as e:
            logger.log_error(f"Error saving analysis results for {file_id}", e)
            raise

    def load_audio_file(self, file_path: str) -> tuple:
        """
        Load an audio file and return its data and sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            data, sample_rate = sf.read(file_path)
            logger.log_file_operation("load", file_path, True)
            return data, sample_rate
            
        except Exception as e:
            logger.log_error(f"Error loading audio file {file_path}", e)
            raise

    def load_transcript(self, file_id: str) -> str:
        """
        Load a transcript file.
        
        Args:
            file_id: File ID of the transcript
            
        Returns:
            str: Transcript text
        """
        try:
            file_path = os.path.join(self.transcript_dir, f"{file_id}_transcript.txt")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            logger.log_file_operation("load", file_path, True)
            return transcript
            
        except Exception as e:
            logger.log_error(f"Error loading transcript for {file_id}", e)
            raise

    def load_analysis_results(self, file_id: str) -> Dict:
        """
        Load analysis results for a file.
        
        Args:
            file_id: File ID of the results
            
        Returns:
            Dict: Analysis results
        """
        try:
            file_path = os.path.join(self.results_dir, f"{file_id}_results.json")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.log_file_operation("load", file_path, True)
            return results
            
        except Exception as e:
            logger.log_error(f"Error loading analysis results for {file_id}", e)
            raise

    def list_audio_files(self) -> List[str]:
        """
        List all audio files in the audio directory.
        
        Returns:
            List[str]: List of audio file paths
        """
        try:
            files = []
            for file in os.listdir(self.audio_dir):
                if os.path.splitext(file)[1].lower() in self.supported_formats:
                    files.append(os.path.join(self.audio_dir, file))
            return files
            
        except Exception as e:
            logger.log_error("Error listing audio files", e)
            raise

    def list_transcripts(self) -> List[str]:
        """
        List all transcript files.
        
        Returns:
            List[str]: List of transcript file paths
        """
        try:
            return [
                os.path.join(self.transcript_dir, f)
                for f in os.listdir(self.transcript_dir)
                if f.endswith('_transcript.txt')
            ]
        except Exception as e:
            logger.log_error("Error listing transcripts", e)
            raise

    def list_results(self) -> List[str]:
        """
        List all results files.
        
        Returns:
            List[str]: List of results file paths
        """
        try:
            return [
                os.path.join(self.results_dir, f)
                for f in os.listdir(self.results_dir)
                if f.endswith('_results.json')
            ]
        except Exception as e:
            logger.log_error("Error listing results", e)
            raise

    def delete_file(self, file_path: str):
        """
        Delete a file.
        
        Args:
            file_path: Path to file to delete
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.log_file_operation("delete", file_path, True)
            else:
                logger.log_warning(f"File not found: {file_path}")
                
        except Exception as e:
            logger.log_error(f"Error deleting file {file_path}", e)
            raise

    def get_file_info(self, file_path: str) -> Dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict: File information
        """
        try:
            stats = os.stat(file_path)
            return {
                'size': stats.st_size,
                'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'format': os.path.splitext(file_path)[1].lower(),
                'path': file_path
            }
        except Exception as e:
            logger.log_error(f"Error getting file info for {file_path}", e)
            raise

# Create a default file handler instance
default_handler = FileHandler()
