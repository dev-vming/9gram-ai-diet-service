import { MailVerificationService } from './mail-verification.service';
import { DataSource } from 'typeorm';
import { VerificationCode } from './entities/verification-code.entity';
import { VerificationCodeRepository } from './repositories/verification-code.repository';
import { UserService } from 'src/user/user.service';
import { Controller, Get, HttpException, HttpStatus, Param, Post, Req, Res, UseFilters, UseGuards, UsePipes, Body } from '@nestjs/common';
import { GoogleAuthGuard } from './utils/guards/google.auth.guard';
import { AuthService } from './auth.service';
import { LocalSignupDto } from './dto/localSignupDto';
import { ApiBody, ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import { LocalAuthGuard } from './utils/guards/local.auth.guard';
import { isLoggedInGuard } from './utils/guards/isLoggedin.guard';
import { LocalLoginDto } from './dto/localLoginDto';
import { HealthInfo } from 'src/user/entities/health-info.entity';
import { User } from 'src/user/entities/user.entity';
import * as dotenv from 'dotenv';
import { isNotLoggedInGuard } from './utils/guards/isNotLoggedIn.guard';
import { LocalLoginDtoValidationPipe } from './utils/pipes/localLoginDtoValidation.pipe';
import { VerifyCodeDto } from './dto/verifyCode.dto';
import { sendVerificationCodeDto } from './dto/sendVerificationCode.dto';
import { SendVoiceMessageRequest } from 'aws-sdk/clients/pinpointsmsvoice';
dotenv.config();


@ApiTags('auth')
@Controller('auth')
export class AuthController {

    constructor(
        private readonly authService: AuthService,
        private readonly userService: UserService,
        private readonly MailVerificationService: MailVerificationService,
    ) {}

    
    /* 구글 로그인 및 사용자등록 */
    @ApiOperation({ summary: '구글 로그인 요청 API' })
    @UseGuards(GoogleAuthGuard)
    @UseGuards(isNotLoggedInGuard)
    @Get('google/login')
    handleGoogleLogin() {}

    /* 구글 로그인 후 리다이렉트되는 URL*/
    @ApiOperation({ summary: '구글서버에서 사용자인증 후 accessToken을 보내는 주소'})
    @UseGuards(GoogleAuthGuard) // 여기서 req.user에 user 정보가 담김
    @UseGuards(isNotLoggedInGuard)
    @Get('google/redirect')
    async handleGoogleRedirect(@Req() request:any ,@Res() response:any) {
        console.log('구글로그인 진행중...')
        // response.status(200).redirect(`${process.env.CLIENT_BASE_URL}/home`);
        response.status(200).redirect(`${process.env.CLIENT_BASE_URL}/auth/google`);
        // const user: User & HealthInfo = await this.userService.getUserInfos(request.user.userId);
        // if(user.recentHealthInfoId === null){
        //     response.status(200).redirect(`${process.env.CLIENT_BASE_URL}/onboarding/1`);
        // }else{
        //     response.status(200).redirect(`${process.env.CLIENT_BASE_URL}/home`);
        // }
    }

    /* 로컬 회원가입 */
    @ApiOperation({ summary: '로컬 회원가입 요청 API' })
    @ApiBody({ type: LocalSignupDto, description: '회원가입 정보'})
    @UseGuards(isNotLoggedInGuard)
    @Post('local/signup')
    async handleLocalSignup(@Body() localSignupDto: LocalSignupDto, @Res() response: any) {
        console.log('회원가입진행중...', localSignupDto);
        await this.authService.localSignup(localSignupDto);
        response.status(200).send('회원가입 성공'); 
    }
    /* 로컬 로그인 */
    @ApiOperation({ summary: '로컬 로그인 요청 API' })
    @ApiBody({ type: LocalLoginDto, description: '로그인 정보'})
    @Post('local/login')
    @UseGuards(isNotLoggedInGuard, LocalAuthGuard)
    async handleLocalLogin(
        @Req() request: any, @Res() response: any){
        console.log('로그인 진행중...', request.user);
        const user = await this.userService.getUserInfos(request.user.userId);
        response.status(200).send(user);
    }

    /* 구글, 로컬 공용 로그아웃 API */
    @ApiOperation({ summary: '로그아웃 uri' })
    @Get('logout')
    async handleGoogleLogout(@Req() request: any, @Res() response: any): Promise<void> {

        try{
            const sessionId = request.signedCookies[process.env.SESSION_COOKIE_NAME];
            if(!sessionId || !request.user) throw new HttpException('로그아웃된 상태입니다.', HttpStatus.CONFLICT);
            // console.log('로그아웃 후의 세션저장소 :', request.sessionStore,'\n로그아웃 힐 세션아이디: ', sessionId );
            await request.sessionStore.destroy(sessionId, (err)=>err && new HttpException('로그아웃 실패',err));
            delete request.user;
            await response.clearCookie(process.env.SESSION_COOKIE_NAME, {signed: true});
            response.status(200).send('로그아웃 성공');
            console.log(`${sessionId}가 로그아웃 되었습니다.`)

        }catch(err){ 
            if(err.message) throw err;
            else throw new HttpException('로그아웃 실패',HttpStatus.BAD_REQUEST); 
        }
        // finally{ console.log('로그아웃 후의 세션저장소: ', request.sessionStore); }
    }

    /* 구글, 로컬 공용 회원 탈퇴 API */
    @ApiOperation({ summary: '회원 탈퇴 API' })
    @Get('withdrawal')
    async handleWithdrawal(@Req() request: any, @Res() response: any): Promise<void> {
        try{
            // 회원 탈퇴
            if(!request.user) throw new HttpException('로그아웃된 상태입니다.', HttpStatus.CONFLICT);
            const result = await this.authService.withdrawal(request.user.userId);
            console.log(result);

            // 세션, req.user, cookie 삭제
            const sessionId = request.signedCookies[process.env.SESSION_COOKIE_NAME];
            if(!sessionId) throw new HttpException('로그아웃된 상태입니다.', HttpStatus.CONFLICT);
            await request.sessionStore.destroy(sessionId, (err)=>err && {msg: '회원탈퇴 실패', err: err});
            delete request.user;
            await response.clearCookie(process.env.SESSION_COOKIE_NAME, {signed: true});
            response.status(200).send(result);
        }catch(err){
            if(err.message) throw err;
            else throw new HttpException('회원탈퇴 실패',HttpStatus.BAD_REQUEST); 
        }
    }

    /* 이메일 인증 코드 발송하기 */
    @ApiOperation({ summary: '이메일 인증 코드 발송하기' })
    @ApiBody({type: sendVerificationCodeDto})
    @Post('/verify-email/send-code')
    async handleSendVerificationCode(@Body() sendVerificationCodeDto:sendVerificationCodeDto, @Req() request: any, @Res() response: any): Promise<void> {
        try{
            const { email } = sendVerificationCodeDto;
            await this.userService.checkEmail(email);
            await this.MailVerificationService.sendVerificationCode(email);
            response.status(200).send('이메일 인증 코드 발송 성공');
        }catch(err){ throw err; }
    }

    /* 인증코드 검증하기 */
    @ApiOperation({ summary: '인증코드 검증하기' })
    @ApiBody({type: VerifyCodeDto})
    @Post('/verify-email/check-code')
    async handleVerifyCode(@Body() body:VerifyCodeDto, @Res() response: any): Promise<void> {
        try{
            const { email, code } = body;
            let verified = await this.MailVerificationService.verifyCode(email, code.toString());
            response.status(200).send({verified});
        }catch(err){ throw err; }
    }
    }
