# Procedimiento de Gestión de Incidentes de Seguridad - TechCorp

Este procedimiento describe cómo se reporta, clasifica, atiende y cierra un incidente
de seguridad de la información en TechCorp. Complementa a la Política de Seguridad
Informática: la notificación inicial por parte de cualquier empleado debe hacerse en
el plazo que fija la Política de Seguridad Informática, y a partir de ese aviso rige el
circuito que se detalla a continuación. Lo administra el área de Seguridad de la
Información y aplica a empleados, contratistas y proveedores con acceso a sistemas
corporativos.

## Qué se considera un incidente

Un incidente de seguridad es cualquier evento que comprometa, o pueda comprometer, la
confidencialidad, integridad o disponibilidad de la información o de los servicios de
la empresa. Algunos ejemplos son: correos de phishing que hayan sido abiertos, accesos
no autorizados a cuentas, pérdida o robo de equipos corporativos, presencia de
software malicioso, exposición accidental de datos de clientes y caídas de servicios
provocadas por un ataque. Ante la duda, siempre conviene reportar: un falso positivo
no tiene ninguna consecuencia para quien lo informa.

## Canales de reporte

Los incidentes se reportan por el canal #incidentes-seguridad de la mensajería interna
o mediante el formulario de incidentes del portal interno. El reporte debe incluir qué
se observó, cuándo, en qué equipo o sistema, y qué acciones se tomaron hasta ese
momento. No se deben reenviar correos sospechosos a otros compañeros ni intentar
investigar por cuenta propia, porque eso puede alterar la evidencia o propagar el
problema.

## Clasificación por severidad y tiempos de respuesta

Una vez recibido el reporte, el equipo de guardia de Seguridad de la Información
clasifica el incidente en uno de cuatro niveles. Cada nivel tiene un tiempo máximo de
primera respuesta, contado desde la recepción del reporte:

- SEV1: servicio crítico caído o fuga de datos confirmada. Primera respuesta en 15
  minutos, con convocatoria inmediata del equipo de respuesta.
- SEV2: degradación severa de un servicio o compromiso de una cuenta con privilegios.
  Primera respuesta en 1 hora.
- SEV3: impacto acotado, por ejemplo un equipo individual con software malicioso
  contenido. Primera respuesta en 4 horas hábiles.
- SEV4: consulta o evento menor sin impacto comprobado, como un correo sospechoso que
  no fue abierto. Primera respuesta en 2 días hábiles.

La severidad puede subir o bajar durante la investigación a medida que se obtiene más
información; cada cambio se registra en el ticket del incidente con su justificación.

## Roles durante la respuesta

En incidentes SEV1 y SEV2, Seguridad de la Información asigna un Incident Commander,
que coordina a los equipos técnicos, prioriza las acciones de contención y toma las
decisiones operativas hasta el cierre. El Incident Commander designa además a un
responsable de comunicaciones, encargado de informar el estado a la dirección, a las
áreas afectadas y, cuando corresponde, a clientes y organismos externos. Los equipos
técnicos ejecutan las tareas de contención y recuperación y documentan en el ticket
cada acción realizada, con su horario.

## Preservación de evidencia

Si se sospecha que un equipo está comprometido, no se debe apagar: el apagado borra la
memoria y con ella información valiosa para la investigación. Lo correcto es
desconectarlo de la red, ya sea quitando el cable o desactivando la conexión
inalámbrica, y dejarlo encendido hasta que el equipo de Seguridad indique cómo
continuar. Tampoco se deben borrar archivos, registros ni mensajes relacionados con el
evento.

## Comunicación externa

Cuando un incidente SEV1 involucra datos de clientes afectados, la empresa comunica la
situación a los clientes dentro de las 24 horas posteriores a la confirmación. La
comunicación la redacta el responsable de comunicaciones junto con el área Legal y
debe describir qué ocurrió, qué datos se vieron involucrados y qué medidas se están
tomando. Ningún empleado está autorizado a comentar el incidente con terceros por fuera
de este circuito.

## Cierre y postmortem

Un incidente se cierra cuando la amenaza fue contenida, los servicios se restablecieron
y se aplicaron las correcciones inmediatas. Para todos los incidentes SEV1 y SEV2 es
obligatorio realizar un postmortem sin culpables dentro de los 5 días hábiles
posteriores al cierre. El documento describe la cronología, la causa raíz, qué
funcionó bien, qué se puede mejorar y las acciones preventivas, cada una con un
responsable y una fecha comprometida. Los postmortem se publican en el espacio interno
de Seguridad para que todas las áreas puedan aprender de lo ocurrido.
