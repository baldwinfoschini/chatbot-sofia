"""
generar_reglamento.py - Genera el PDF del Reglamento Estudiantil FICTICIO usado como corpus del chatbot.

El documento corresponde a una institución inventada ("Institución Educativa Virtual Horizonte")
y fue elaborado con fines académicos para el ACA 2 de Computación en la Nube (CUN).
Uso: python herramientas/generar_reglamento.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

SALIDA = Path(__file__).resolve().parents[1] / "docs" / "Reglamento_Estudiantil_IE_Virtual_Horizonte.pdf"
INSTITUCION = "Institución Educativa Virtual Horizonte"

REGLAMENTO = [
    ("TÍTULO I. DISPOSICIONES GENERALES", [
        ("Objeto", "El presente reglamento establece los derechos, deberes, procesos académicos y normas de convivencia "
         "de los estudiantes de la Institución Educativa Virtual Horizonte, en adelante la Institución."),
        ("Ámbito de aplicación", "Se aplica a todos los estudiantes matriculados en los niveles de primaria (grados 1 a 5), "
         "secundaria (grados 6 a 8) y bachillerato (grados 9 a 12), en la modalidad virtual que se desarrolla en la "
         "plataforma de aprendizaje institucional."),
        ("Principios", "La Institución se rige por los principios de inclusión, equidad, acompañamiento oportuno, "
         "evaluación formativa, respeto, corresponsabilidad de la familia y uso ético de la información."),
        ("Plataforma de aprendizaje", "Las clases, recursos, anuncios, foros de discusión y evaluaciones se gestionan en la "
         "plataforma institucional. La plataforma registra la participación del estudiante (intervenciones en clase, "
         "recursos consultados, anuncios leídos y aportes en foros), información que se usa para la evaluación y el "
         "acompañamiento según lo dispuesto en este reglamento."),
    ]),
    ("TÍTULO II. DERECHOS Y DEBERES DE LOS ESTUDIANTES", [
        ("Derechos", "Son derechos del estudiante: a) recibir una educación de calidad y un trato respetuoso; "
         "b) conocer al inicio de cada semestre los criterios de evaluación de cada asignatura; c) recibir sus "
         "calificaciones en un plazo máximo de 8 días hábiles después de cada evaluación; d) presentar reclamos sobre sus "
         "calificaciones; e) acceder a tutorías, orientación y servicios de bienestar; f) ser informado cuando se active "
         "una alerta temprana sobre su desempeño; g) solicitar la homologación de asignaturas cursadas en otra institución; "
         "h) la protección de sus datos personales."),
        ("Deberes", "Son deberes del estudiante: a) participar activamente en las clases, levantando la mano y aportando "
         "en las sesiones; b) consultar los recursos del curso y leer los anuncios publicados por los docentes; "
         "c) participar en los foros de discusión con respeto; d) asistir a las sesiones programadas y justificar sus "
         "ausencias; e) entregar las actividades en las fechas establecidas; f) actuar con honestidad académica; "
         "g) cumplir las normas de convivencia digital."),
    ]),
    ("TÍTULO III. MATRÍCULA Y PERMANENCIA", [
        ("Matrícula", "La matrícula se realiza por semestre académico. El primer semestre inicia en febrero y el segundo "
         "en agosto. Para matricularse se requiere el formulario de inscripción, el documento de identidad, el certificado "
         "del último grado aprobado y el registro del padre, madre o acudiente responsable."),
        ("Requisitos de permanencia", "Para conservar la calidad de estudiante se requiere: a) aprobar como mínimo el 75 % "
         "de las asignaturas matriculadas en el semestre; b) no superar 15 ausencias injustificadas en el semestre; "
         "c) cumplir el plan de acompañamiento cuando se le haya asignado; d) no haber sido sancionado con falta gravísima."),
        ("Periodo de prueba académica", "El estudiante que repruebe entre el 25 % y el 50 % de las asignaturas del semestre "
         "entra en periodo de prueba académica durante el semestre siguiente. En ese periodo debe asistir a tutorías "
         "semanales y su familia debe firmar un compromiso de acompañamiento."),
        ("Pérdida de la calidad de estudiante", "Se pierde la calidad de estudiante cuando: a) se reprueba más del 50 % de las "
         "asignaturas en un semestre; b) se reprueba el periodo de prueba académica; c) se cancela voluntariamente la "
         "matrícula; d) se impone la sanción de cancelación de matrícula por falta gravísima."),
        ("Reintegro", "Quien haya perdido la calidad de estudiante por razones académicas puede solicitar el reintegro una "
         "sola vez, después de un semestre por fuera de la Institución, presentando un plan de mejoramiento aprobado por "
         "Coordinación Académica."),
    ]),
    ("TÍTULO IV. EVALUACIÓN ACADÉMICA", [
        ("Escala de valoración", "Las asignaturas se califican de 0 a 100 puntos. La nota final se expresa en tres niveles "
         "de desempeño: nivel bajo (L), de 0 a 69 puntos; nivel medio (M), de 70 a 89 puntos; y nivel alto (H), de 90 a 100 "
         "puntos. La asignatura se aprueba con 70 puntos o más."),
        ("Componentes de la nota", "La nota final de cada asignatura se compone de: evaluaciones y trabajos, 50 %; "
         "participación en clase, 15 %; consulta de recursos y anuncios del curso, 15 %; y participación en foros de "
         "discusión, 20 %. El docente puede ajustar estos porcentajes hasta en 10 puntos, informándolo el primer día de clases."),
        ("Semestres y cortes", "Cada semestre tiene dos cortes de evaluación. El primer corte vale el 40 % y el segundo "
         "el 60 % de la nota final. Las notas de cada corte se publican en la plataforma."),
        ("Actividades de recuperación", "El estudiante con nivel bajo (nota final entre 50 y 69) en una asignatura puede "
         "presentar una actividad de recuperación dentro de los 15 días hábiles siguientes a la publicación de la nota. "
         "Si la aprueba, la nota final de la asignatura será de 70 puntos. Con nota inferior a 50 debe repetir la asignatura."),
        ("Reclamos sobre calificaciones", "El estudiante, o su acudiente, puede presentar un reclamo escrito ante el docente "
         "dentro de los 5 días hábiles siguientes a la publicación de la nota. El docente responde en 5 días hábiles. "
         "Si no está de acuerdo con la respuesta, puede apelar ante Coordinación Académica en los 3 días hábiles siguientes."),
        ("Honestidad académica", "El plagio, la copia, la suplantación o el uso no autorizado de herramientas de "
         "inteligencia artificial en evaluaciones anula la actividad con nota de 0 y se trata como falta grave."),
    ]),
    ("TÍTULO V. ASISTENCIA Y AUSENCIAS", [
        ("Registro de asistencia", "La asistencia se registra en cada sesión sincrónica. Se considera ausencia no conectarse "
         "a la sesión o conectarse después de transcurrido el 50 % de su duración."),
        ("Límites de ausencias", "Cuando un estudiante acumula más de 7 ausencias en el semestre, se activa una alerta "
         "temprana y se informa al acudiente. Con más de 15 ausencias injustificadas en una asignatura, el estudiante la "
         "reprueba con nota de 0, sin derecho a recuperación."),
        ("Justificación de ausencias", "Las ausencias se justifican dentro de los 3 días hábiles siguientes, mediante "
         "excusa médica, calamidad doméstica comprobada o representación institucional. Las ausencias justificadas no "
         "cuentan para el límite de reprobación, pero sí se tienen en cuenta para el acompañamiento."),
    ]),
    ("TÍTULO VI. SISTEMA DE ALERTAS TEMPRANAS Y ACOMPAÑAMIENTO", [
        ("Indicadores de alerta temprana", "La Institución activa una alerta temprana cuando se presenta al menos una de "
         "estas situaciones: a) más de 7 ausencias en el semestre; b) nota de nivel bajo en el primer corte; c) menos de 20 "
         "recursos del curso consultados en el corte; d) participación en clase muy baja según el registro de la plataforma; "
         "e) el acudiente no respondió la encuesta institucional ni asistió a la reunión de inicio de semestre."),
        ("Plan de acompañamiento", "Activada la alerta, el director de grupo se reúne con el estudiante y su acudiente dentro "
         "de los 10 días hábiles siguientes y acuerda un plan de acompañamiento que puede incluir tutorías, refuerzo en las "
         "asignaturas en riesgo, orientación psicológica y seguimiento quincenal de la participación en la plataforma."),
        ("Uso de modelos predictivos", "La Institución puede usar modelos estadísticos o de inteligencia artificial para "
         "identificar estudiantes en riesgo a partir de los datos de la plataforma. Estos modelos son una herramienta de "
         "apoyo: ninguna decisión académica o disciplinaria se toma solo con base en su resultado, y toda alerta es revisada "
         "por un docente u orientador."),
    ]),
    ("TÍTULO VII. PARTICIPACIÓN DE LA FAMILIA", [
        ("Acudiente responsable", "Cada estudiante debe tener registrado un acudiente responsable (padre, madre u otro "
         "adulto autorizado), quien recibe las comunicaciones de la Institución y acompaña el proceso académico."),
        ("Encuesta institucional", "Cada semestre la Institución aplica una encuesta a los acudientes sobre el proceso "
         "académico y su satisfacción con la Institución. Responderla es un deber del acudiente, y sus resultados se usan "
         "para mejorar el servicio."),
        ("Reuniones con la familia", "Se realizan al menos dos reuniones generales por semestre y las reuniones individuales "
         "que requiera el plan de acompañamiento. El acudiente puede solicitar una reunión con el director de grupo en "
         "cualquier momento."),
    ]),
    ("TÍTULO VIII. HOMOLOGACIÓN, TRASLADOS Y ESTUDIANTES EXTRANJEROS", [
        ("Homologación de asignaturas", "El estudiante que haya cursado asignaturas en otra institución puede solicitar su "
         "homologación ante Coordinación Académica, como mínimo 30 días antes del inicio del semestre. Requisitos: "
         "a) certificado de notas oficial; b) contenidos del programa de la asignatura; c) equivalencia de contenidos de al "
         "menos el 80 %; d) nota aprobatoria igual o superior a 70 sobre 100, o su equivalente. La respuesta se entrega en "
         "un plazo máximo de 15 días hábiles."),
        ("Traslado de sección", "El estudiante puede solicitar el traslado entre las secciones A, B y C de su grado en las "
         "dos primeras semanas del semestre, siempre que haya cupo. El traslado no modifica las notas ya registradas."),
        ("Estudiantes extranjeros", "Los estudiantes que provengan de otro país deben presentar la convalidación de sus "
         "estudios ante la autoridad educativa nacional. Mientras se resuelve, pueden matricularse de forma provisional "
         "por un semestre."),
    ]),
    ("TÍTULO IX. CONVIVENCIA Y CONDUCTA DIGITAL", [
        ("Normas de convivencia digital", "En las clases y foros, el estudiante debe usar un lenguaje respetuoso, mantener "
         "la cámara y el micrófono según lo indique el docente, no compartir enlaces o contenidos ajenos a la clase y "
         "respetar la privacidad de compañeros y docentes."),
        ("Clasificación de las faltas", "Las faltas se clasifican en: a) leves, como interrumpir la clase o usar el chat "
         "para fines ajenos; b) graves, como el plagio, la suplantación o el irrespeto reiterado; c) gravísimas, como el "
         "acoso, la difusión de contenido íntimo o la alteración de calificaciones."),
        ("Sanciones y debido proceso", "Las sanciones son: amonestación escrita (faltas leves), suspensión de 1 a 5 días "
         "(faltas graves) y cancelación de la matrícula (faltas gravísimas). Antes de sancionar se notifica al estudiante "
         "y a su acudiente, quienes tienen 5 días hábiles para presentar descargos. Las sanciones pueden apelarse ante el "
         "Consejo Directivo."),
    ]),
    ("TÍTULO X. BIENESTAR Y ESTÍMULOS", [
        ("Servicios de bienestar", "La Institución ofrece orientación psicológica, tutorías académicas, talleres de hábitos "
         "de estudio y apoyo para el uso de la plataforma. Los servicios se solicitan a través del módulo de bienestar."),
        ("Estímulos al rendimiento", "Los estudiantes con nivel alto (90 puntos o más) en el promedio del semestre ingresan "
         "al cuadro de honor. El mejor promedio de cada grado recibe una beca del 50 % del valor de la matrícula del "
         "semestre siguiente."),
    ]),
    ("TÍTULO XI. PROTECCIÓN DE DATOS PERSONALES", [
        ("Tratamiento de datos", "Los datos personales y de actividad en la plataforma se usan solo para fines académicos, "
         "de acompañamiento y de mejoramiento institucional. No se comparten con terceros sin autorización del acudiente, "
         "salvo obligación legal."),
        ("Derechos del titular", "El estudiante y su acudiente pueden consultar, actualizar o solicitar la corrección de "
         "sus datos a través de la Secretaría Académica, que responde en un plazo máximo de 10 días hábiles."),
    ]),
    ("DISPOSICIONES FINALES", [
        ("Vigencia", "El presente reglamento rige a partir de su publicación y deroga las disposiciones anteriores que le "
         "sean contrarias. Cualquier modificación debe ser aprobada por el Consejo Directivo."),
    ]),
]


def pie(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(2 * cm, 1.2 * cm, f"Reglamento Estudiantil – {INSTITUCION} (institución ficticia, uso académico)")
    canvas.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.restoreState()


def main():
    SALIDA.parent.mkdir(exist_ok=True)
    ss = getSampleStyleSheet()
    titulo = ParagraphStyle("t", parent=ss["Title"], fontSize=20, leading=24, textColor=colors.HexColor("#0F2747"))
    sub = ParagraphStyle("s", parent=ss["Normal"], alignment=TA_CENTER, fontSize=11, leading=14,
                         textColor=colors.HexColor("#333333"))
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=12.5, leading=16, spaceBefore=14, spaceAfter=6,
                       textColor=colors.HexColor("#0F2747"))
    cuerpo = ParagraphStyle("c", parent=ss["Normal"], fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7)
    aviso = ParagraphStyle("a", parent=cuerpo, fontSize=9.5, leading=13, textColor=colors.HexColor("#5A3D00"))

    story = [Paragraph("REGLAMENTO ESTUDIANTIL", titulo), Paragraph(INSTITUCION, sub),
             Paragraph("Primaria, secundaria y bachillerato – Modalidad virtual", sub), Spacer(1, 12)]
    caja = Table([[Paragraph("<b>Documento ficticio.</b> Este reglamento fue elaborado con fines exclusivamente académicos "
                             "para el ACA 2 de Computación en la Nube (CUN). La Institución Educativa Virtual Horizonte no "
                             "existe y este texto no corresponde a la normativa de ninguna institución real.", aviso)]],
                 colWidths=[16.5 * cm])
    caja.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDF0D2")),
                              ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#F3A712")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                              ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [caja, Spacer(1, 10)]
    n = 0
    for titulo_sec, articulos in REGLAMENTO:
        bloque = [Paragraph(titulo_sec, h)]
        for nombre, texto in articulos:
            n += 1
            p = Paragraph(f"<b>Artículo {n}. {nombre}.</b> {texto}", cuerpo)
            if bloque:
                story.append(KeepTogether(bloque + [p]))
                bloque = []
            else:
                story.append(p)
    SimpleDocTemplate(str(SALIDA), pagesize=letter, leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.2 * cm,
                      bottomMargin=2.2 * cm, title="Reglamento Estudiantil – IE Virtual Horizonte (ficticio)",
                      author="Equipo ACA 2 Computación en la Nube").build(story, onFirstPage=pie, onLaterPages=pie)
    print(f"PDF generado: {SALIDA} ({n} artículos)")


if __name__ == "__main__":
    main()
