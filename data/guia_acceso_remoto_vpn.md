# Guía de Acceso Remoto (VPN) - TechCorp

Esta guía explica cómo conectarse de forma segura a los sistemas internos de TechCorp
cuando no se está físicamente en una de nuestras oficinas. Está dirigida a todas las
personas que trabajan en la empresa, incluidas las que tienen un esquema híbrido, y
a los equipos de proveedores que cuenten con una cuenta corporativa habilitada. La
mantiene la Mesa de Ayuda IT y se revisa cada vez que se publica una nueva versión del
cliente de conexión.

## Alcance y cliente obligatorio

El cliente VPN corporativo TC-Connect es obligatorio para acceder a cualquier sistema
interno de TechCorp fuera de la oficina: repositorios de código, intranet, herramientas
de gestión, tableros de monitoreo y carpetas compartidas. La obligación incluye los
días de teletrabajo, los viajes de negocios y cualquier conexión desde un domicilio
particular. No existe ningún otro método autorizado de acceso remoto.

Se debe utilizar TC-Connect en su versión 5.2 o superior. Las versiones anteriores
quedan bloqueadas automáticamente por el gateway, porque no soportan los algoritmos
de cifrado vigentes. El cliente se descarga desde el portal de autoservicio de IT,
ubicado en portal.techcorp.example, y se actualiza solo cuando la Mesa de Ayuda publica
una nueva versión. Si el cliente muestra un aviso de actualización pendiente, conviene
instalarla antes de terminar la jornada para evitar cortes al día siguiente.

## Autenticación y duración de la sesión

Para iniciar una sesión se ingresa el usuario corporativo, la contraseña y un código de
2FA generado por la aplicación autenticadora registrada durante el alta. El código
cambia cada 30 segundos y no puede reutilizarse. Nunca se debe compartir el código ni
aprobar solicitudes de autenticación que uno no haya iniciado; ante una solicitud
inesperada, rechazarla y avisar a la Mesa de Ayuda.

Por razones de seguridad, la sesión expira a las 12 horas de haberse iniciado, aunque
haya actividad. Al expirar, TC-Connect cierra el túnel y pide autenticarse de nuevo. Se
recomienda guardar el trabajo en curso antes de que se cumpla ese plazo, ya que las
conexiones a bases de datos y escritorios remotos se interrumpen al cerrarse el túnel.

## Configuración de red y dispositivos permitidos

En TechCorp el split tunneling está deshabilitado: mientras TC-Connect está conectado,
todo el tráfico del equipo pasa por la VPN, incluida la navegación general. Esto permite
aplicar los mismos controles de filtrado y monitoreo que en la oficina. Por ese motivo
algunos servicios de streaming pueden funcionar más lento durante la conexión, lo cual
es esperable y no constituye una falla.

Está prohibido conectarse desde dispositivos personales que no tengan instalado el
perfil MDM provisto por IT. Los equipos corporativos ya vienen configurados y no
requieren pasos adicionales. Quien necesite usar un equipo propio debe solicitar la
instalación del perfil MDM a través del portal de autoservicio antes del primer
intento de conexión; sin ese perfil, el gateway rechaza el equipo aunque las
credenciales sean correctas.

Tampoco está permitido instalar otros clientes de túnel o proxies en el mismo equipo,
porque interfieren con TC-Connect y generan rutas de red inconsistentes.

## Errores frecuentes

La mayoría de los problemas de conexión se resuelven sin intervención de soporte. Los
códigos de error que muestra el cliente son los siguientes:

- VPN-401: credenciales inválidas o código 2FA vencido. Verificar el usuario y la
  contraseña y, si el error persiste, resincronizar la app autenticadora desde su menú
  de configuración, porque la hora del teléfono puede estar desfasada.
- VPN-503: el gateway principal está saturado. Reintentar en 5 minutos o, si es urgente,
  configurar en TC-Connect el gateway secundario vpn2.techcorp.internal, que tiene la
  misma configuración y atiende el tráfico excedente.
- VPN-809: el puerto que usa la VPN está bloqueado por la red local, algo habitual en
  hoteles, aeropuertos y redes públicas. Se resuelve activando el modo TCP 443 en
  TC-Connect (menú Preferencias, opción Conexión), que encapsula el túnel en el mismo
  puerto que usa la navegación segura.

Si aparece un código distinto de los anteriores, se recomienda tomar una captura de
pantalla del mensaje completo y adjuntarla al reporte.

## Buenas prácticas de uso

Desconectar la VPN al terminar la jornada reduce la carga del gateway y evita sesiones
abiertas en equipos desatendidos. Antes de viajar, probar la conexión desde casa con el
modo TCP 443 activado ayuda a detectar problemas con anticipación. En redes públicas se
desaconseja trabajar con información confidencial aun con la VPN activa si la pantalla
queda visible para terceros. Ante la pérdida o el robo de un equipo con TC-Connect
instalado, avisar de inmediato a la Mesa de Ayuda para revocar sus certificados.

## Soporte

Las consultas sobre acceso remoto se canalizan por la Mesa de Ayuda IT, en el interno
4357 o por el portal de autoservicio. La atención telefónica funciona de lunes a viernes
de 8 a 20 hs. Fuera de ese horario solo se atienden incidentes críticos que impidan la
operación de un servicio esencial; las consultas de configuración o de instalación
quedan registradas y se responden el siguiente día hábil.
