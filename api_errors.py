"""Stable public error identifiers; technical exceptions stay in server logs."""
import re


class PublicError(ValueError):
    def __init__(self, code, message, *, status=400, details=None):
        super().__init__(message)
        self.code, self.status, self.details = code, status, details or {}


def error_info(exc, message=None):
    message = message or str(exc)
    code, details = getattr(exc, 'code', None), getattr(exc, 'details', {})
    if not code:
        unfinished = re.match(r'Image (\d+) is unfinished\.', message)
        if unfinished:
            code, details = 'ANNOTATION_UNFINISHED', {'imageId': int(unfinished[1])}
        elif message.startswith('Classification images require'):
            code = 'CLASSIFICATION_LABEL_REQUIRED'
        elif message.startswith('At least one submitted training image'):
            code = 'TRAIN_IMAGES_REQUIRED'
        elif message.startswith('Preview expired'):
            code = 'PREVIEW_EXPIRED'
        elif 'unavailable' in message.lower():
            code = 'SERVICE_UNAVAILABLE'
        elif isinstance(exc, (FileNotFoundError,)):
            code = 'NOT_FOUND'
        elif isinstance(exc, FileExistsError):
            code = 'CONFLICT'
        elif isinstance(exc, ValueError):
            code = 'VALIDATION_ERROR'
        else:
            code = 'OPERATION_FAILED'
    return {'code': code, 'details': details}
