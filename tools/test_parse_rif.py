import json
import re
import html as _html_lib


def _html_to_text(html_text: str) -> str:
	if not html_text:
		return ""
	try:
		s = _html_lib.unescape(html_text)
	except Exception:
		s = html_text
	s = re.sub(r"(?i)<br\s*/?>", "\n", s)
	s = re.sub(r"(?i)</tr>", "\n", s)
	s = re.sub(r"(?i)</td>", "\n", s)
	s = re.sub(r"(?i)</p>", "\n", s)
	s = re.sub(r"<[^>]+>", "", s)
	s = s.replace('\xa0', ' ')
	s = re.sub(r"[ \t\x0b\f\r]+", " ", s)
	lines = [ln.strip() for ln in s.splitlines()]
	lines = [ln for ln in lines if ln]
	return "\n".join(lines)


def _parse_rif_html(html_text: str) -> dict:
	text = _html_to_text(html_text)
	if not text:
		return {}
	out = {"raw_text": text}
	m = re.search(r"COMPROBANTE[^A-Z0-9\n\r\-]*([A-Z0-9\-]+)", text, flags=re.IGNORECASE)
	if m:
		out["numero_comprobante"] = m.group(1).strip()
	rif_match = re.search(r"\b([VEJPG])[-\s]?(\d{6,9})(?:[-\s]?(\d))?\b", text, flags=re.IGNORECASE)
	if rif_match:
		parts = [rif_match.group(1).upper(), rif_match.group(2)]
		if rif_match.group(3):
			parts.append(rif_match.group(3))
		rif_val = "-".join(parts)
		out["rif"] = rif_val
		for line in text.splitlines():
			if rif_match.group(2) in line:
				line_un = line
				try:
					idx = line_un.find(rif_match.group(0))
				except Exception:
					idx = -1
				if idx != -1:
					name_part = line_un[idx + len(rif_match.group(0)):].strip()
					if name_part:
						out["nombre"] = name_part
				break
	m_dom = re.search(r"DOMICILIO\s+FISCAL\s*(.*?)(?:ZONA\s+POSTAL|ZONA\s+POSTAL|ZONA|$)", text, flags=re.IGNORECASE | re.DOTALL)
	if m_dom:
		dom = m_dom.group(1).strip()
		dom = re.sub(r"\s+", " ", dom)
		out["domicilio"] = dom
	m_zp = re.search(r"ZONA\s+POSTAL\s*[:\s]*([0-9]{3,6})", text, flags=re.IGNORECASE)
	if m_zp:
		out["zona_postal"] = m_zp.group(1)
	ger_line = None
	for line in text.splitlines():
		if 'GERENCIA' in line.upper() or 'SEDE REGIONAL' in line.upper():
			ger_line = line.strip()
			break
	if ger_line:
		out["gerencia"] = ger_line
	lines = text.splitlines()
	for i, line in enumerate(lines):
		if 'FIRMA AUTORIZADA' in line.upper():
			code = None
			mcode = re.search(r"([0-9]{6,}-[A-Z0-9]{1,6})", line)
			if mcode:
				code = mcode.group(1)
			else:
				if i > 0:
					prev = lines[i-1]
					m2 = re.search(r"([0-9]{6,}-[A-Z0-9]{1,6})", prev)
					if m2:
						code = m2.group(1)
			if code:
				out["firma_autorizada"] = code
			break
	date_pat = r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b"
	def find_date_after(label: str):
		lab = label.lower()
		lo = text.lower()
		idx = lo.find(lab)
		if idx == -1:
			return None
		tail = text[idx:idx+200]
		m = re.search(date_pat, tail)
		if m:
			return m.group(0)
		m2 = re.search(date_pat, text[idx:idx+400])
		if m2:
			return m2.group(0)
		return None
	fi = find_date_after('FECHA DE INSCRIPCI') or find_date_after('FECHA DE INSCRIPCION')
	fu = find_date_after('FECHA DE &Uacute;LTIMA') or find_date_after('FECHA DE ULTIMA') or find_date_after('FECHA DE ÚLTIMA')
	fv = find_date_after('FECHA DE VENCIMIENTO')
	if fi:
		out['fecha_inscripcion'] = fi
	if fu:
		out['fecha_ultima_actualizacion'] = fu
	if fv:
		out['fecha_vencimiento'] = fv
	return out

html = """





<html>
<head>
<title>SENIAT - Servicio Nacional Integrado de Administraci�n Aduanera y Tributaria</title>
<meta http-equiv="Content-Type" content="aplication/pdf; charset=iso-8859-1">
<link rel="stylesheet" href="http://contribuyente.seniat.gob.ve/css/iseniat.css">

</head>

<body bgcolor="#FFFFFF" text="#000000" oncontextmenu="return false;">
<center>
<br><br>
<table width="80%" class="bordeCuadro">
	<tr>
		<td style="border-bottom:solid 1px #000000;"><img src="http://contribuyente.seniat.gob.ve/imagenes/logo-seniat-int.gif" width="240" height="70"></td>
		<td style="border-bottom:solid 1px #000000;" align="right"><b>N� DE COMPROBANTE: &nbsp;202506K0000068423442</b></td>
	</tr>
	<tr>
		<td colspan="2" align="center" class="letrasSmall" style="border-bottom:solid 1px #000000;"><br><b>REGISTRO &Uacute;NICO DE INFORMACI&Oacute;N FISCAL (RIF)</b></br>&nbsp;</td>
	</tr>	
	<tr>
		<td colspan="2" style="border-bottom:solid 1px #000000;">
			<table>
				<tr>
					<td width="60%"  valign="top">
						<br><b>V286927951</b>&nbsp;RODRIGUEZ CONTRERAS YENDER HUMBERTO</br>
						<br><p align="justify"><b>DOMICILIO FISCAL </b>&nbsp;CALLE CAJIGAL CASA  NRO 03 SECTOR LA CUADRA CARACAS DISTRITO CAPITAL&nbsp;
						    ZONA POSTAL &nbsp;1090</p></br>
						
							
							
								<b>Este Contribuyente no posee firmas personales </b>
							
						
					</td>
					<td width="30%" valign="top">
						<br><b>FECHA DE INSCRIPCI&Oacute;N:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b></br>
						<br><b>FECHA DE &Uacute;LTIMA ACTUALIZACI&Oacute;N:</b></br>
						<br><b>FECHA DE VENCIMIENTO:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b></br>
					</td>
					<td width="10%" valign="top">
						<br>29/01/2021</br>
						<br>19/08/2025</br>
						<br>19/08/2028</br>
					</td>					
				</tr>
			</table>
		</td>
	</tr>		
	<tr>
		<td style="border-bottom:solid 1px #000000;"><b>GERENCIA REGIONAL DE TRIBUTOS INTERNOS LIBERTADOR  / SEDE REGIONAL DE TRIBUTOS INTERNOS LIBERTADOR</b></td>
		<td align="center" style="border-bottom:solid 1px #000000;"><b><br>1286927951-ODB</br>
						  FIRMA AUTORIZADA</b><br>&nbsp;
		</td>
	</tr>	
	<tr>
		<td colspan="2">
				
				<br>
				<p align="justify">
				   	
				   La condici�n de este contribuyente requiere la retenci�n del 100% del impuesto causado, salvo que est� exento, no sujeto o demuestre ante el Agente de Retenci�n del IVA que es un contribuyente exonerado.</p> <br>
						
		</td>
	</tr>	
</table>
</center>
</body>
</html>
"""

res = _parse_rif_html(html)
print(json.dumps(res, ensure_ascii=False, indent=2))
