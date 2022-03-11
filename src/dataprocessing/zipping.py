
import io
import zipfile
import os
from fastapi import Response

def zipfiles(filenames):
    zip_filename = "archive.zip"
    filenames =['output.png']
    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")
    with zipfile.ZipFile(zip_filename, 'w') as zip:
        for file in filenames:
            zip.write(file, compress_type=zipfile.ZIP_DEFLATED)
    zip.close()
    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}' })

    return resp